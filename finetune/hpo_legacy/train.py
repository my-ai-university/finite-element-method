# https://github.com/huggingface/peft/blob/main/examples/sft/train.py
from dataclasses import dataclass, field
import json
import os
from peft import get_peft_model, LoraConfig, TaskType
import sys
from transformers import (HfArgumentParser, set_seed,
                          AutoModelForCausalLM, PreTrainedTokenizerFast, DataCollatorForLanguageModeling,
                          BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from typing import Optional
import torch

from finetune.utils.data_utils import get_dataset_text
from finetune.utils.gpu_utils import clear_gpu_cache


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    use_4bit_quantization: Optional[bool] = field(default=False)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    r: Optional[int] = field(default=64)
    target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj")

@dataclass
class DataArguments:
    data_file: Optional[str] = field(default="./data/qa_with_chat_template.csv")
    split_ratio: Optional[float] = field(default=0.1)
    max_seq_length: Optional[int] = field(default=500)

@dataclass
class UtilsArguments:
    trial_output_file: Optional[str] = field(default="./trial_output_file.json")
    trial_number: Optional[int] = field(default=0)


# https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def get_tokenize_fn(tokenizer, max_seq_length):
    def _tokenize(examples):
        encodings = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_seq_length,
            return_tensors="pt",
            stride=10)
        encodings['labels'] = encodings['input_ids'].clone()
        return encodings
    return _tokenize

# Some notes about models choices
#  we have either z2 and z3 from deeepspeed for optimization
#   z2 have supports for bnb quantization
#   z3 should also have according to the following links but somehow does not have supports for llama 3.X multi modality models using zero.init()
#     https://github.com/huggingface/accelerate/issues/1228
#     https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed
#  Given the above constraints and we use multi modality model:
#  if we want to do half-precision finetuning, we can use z2 or z3 with llama 3.2-11B-Vision-Instruct, llama 3.2-90B-Vision-Instruct
#  if we want to do quantized finetuning, we can use z2 or z3 (no zero.init) with llama 3.2-11B-Vision-Instruct, llama 3.2-90B-Vision-Instruct
def get_model_init_fn(args):

    def _model_init():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config if args.use_4bit_quantization else None,
            trust_remote_code=True,
            attn_implementation="eager", # "flash_attention_2" is not for V100 :(
            torch_dtype=torch.bfloat16)
        model.config._name_or_path = args.model_name_or_path
        model.config.use_cache = False

        peft_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(","),
            bias="none",
            task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, peft_config)

        # SFTTrainer in the current version of trl is not compatible with transformers 4.45.3
        # https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L210C1-L217C101
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model
    return _model_init

def get_split_tokenized_dataset(args, seed, tokenizer):
    dataset = get_dataset_text(args.data_file)
    tokenized_dataset = dataset.map(
        get_tokenize_fn(tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"])
    split_dataset = tokenized_dataset.train_test_split(
        test_size=args.split_ratio,
        seed=seed,
        shuffle=True)
    return split_dataset['train'], split_dataset['test']


def main():
    # get args and set seed
    model_args, data_args, training_args, utils_args = HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, UtilsArguments
    )).parse_args_into_dataclasses()
    set_seed(training_args.seed)

    # clear GPU cache
    clear_gpu_cache(training_args.local_rank)

    # tokenizer
    if training_args.local_rank == 0:
        print(f"\nLoading tokenizer from {model_args.model_name_or_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        return_tensors="pt")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # tokenized dataset
    if training_args.local_rank == 0:
        print(f"\nTokenizing and splitting dataset {data_args.data_file}")
    train_dataset, eval_dataset = get_split_tokenized_dataset(
        data_args, training_args.seed, tokenizer)

    # trainer
    if training_args.local_rank == 0:
        print(f"\nPreparing the HF trainer")
    if training_args.local_rank == 0 and model_args.use_4bit_quantization:
        print(f"\nModel will be quantized")

    trainer = Trainer(
        model_init=get_model_init_fn(model_args),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False))

    # train and evaluate
    if training_args.local_rank == 0:
        print(f"\nStart training for trial {utils_args.trial_number}")
    trainer.train()

    if training_args.local_rank == 0:
        print(f"\nEvaluating model for ...")
    eval_metrics = trainer.evaluate()
    with open(utils_args.trial_output_file, 'w') as f:
        trial_output = {
            'trial_number': utils_args.trial_number,
            'loss': eval_metrics['eval_loss']}
        json.dump(trial_output, f, indent=4)

    # Reset for the next trial
    if training_args.local_rank == 0:
        print("\nClearing GPU cache at the end for the next trial")
    del trainer
    clear_gpu_cache(training_args.local_rank)


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()