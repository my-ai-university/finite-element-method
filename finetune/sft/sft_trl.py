import datasets
from datasets import DatasetDict
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import os
from peft import get_peft_model, LoraConfig, TaskType
import sys
import transformers
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (ModelConfig, SFTConfig,
                 SFTTrainer,
                 TrlParser,
                 DataCollatorForCompletionOnlyLM)
from typing import Optional
import torch

from finetune.utils.data_utils import get_dataset_qa, make_conv
from finetune.utils.gpu_utils import clear_gpu_cache
from finetune.utils.eval_utils import FixedPromptEvaluationCallback
from finetune.utils.prompt import (FEM_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT,
                                   FIXED_PROMPT, FIXED_PROMPT_BASE_MODEL_COMPLETION,
                                   FIXED_ELEPHANT_PROMPT, FIXED_ELEPHANT_PROMPT_BASE_MODEL_COMPLETION)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_file: Optional[str] = field(default="./data/qa_with_chat_template.csv")
    split_ratio: Optional[float] = field(default=0.1)

def get_split_dataset(args, seed):
    dataset = get_dataset_qa(args.data_file)
    split_dataset = dataset.train_test_split(
        test_size=args.split_ratio,
        seed=seed,
        shuffle=True)
    return split_dataset['train'], split_dataset['test']


def main():
    # get args and set seed
    model_args, data_args, training_args = TrlParser((
        ModelConfig, DataConfig, SFTConfig)).parse_args_and_config()
    set_seed(training_args.seed)

    #############
    # Preparation
    #############

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")

    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    logging_dir = os.environ["LOGGING_DIR"]
    output_dir = os.environ["OUTPUT_DIR"]
    training_args.output_dir = f"{output_dir}/{training_args.run_name}"
    training_args.logging_dir = f"{logging_dir}/{training_args.run_name}"
    training_args.run_name = training_args.run_name + f"_{formatted_datetime}"

    #######################################################################
    # Load and preprocess dataset (tokenization is handled by SFT Trainer)
    #######################################################################

    logger.info(f"\nSplitting and chat templating the dataset {data_args.data_file}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
        # padding_side="left",
        return_tensors="pt")
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = "<|reserved_special_token_5|>" # or "<|finetune_right_pad_id|>" https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/tokenizer_config.json

    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|start_header_id|>user<|end_header_id|>",
        response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        tokenizer=tokenizer,
        mlm=False)

    train_dataset, eval_dataset = get_split_dataset(
        data_args, training_args.seed)
    train_dataset = train_dataset.map(
        make_conv,
        fn_kwargs={
            "tokenizer": tokenizer,
            "system_prompt": DEFAULT_SYSTEM_PROMPT if "elephant" in training_args.run_name else FEM_SYSTEM_PROMPT,
        },
        batched=True)
    eval_dataset = eval_dataset.map(
        make_conv,
        fn_kwargs={
            "tokenizer": tokenizer,
            "system_prompt": DEFAULT_SYSTEM_PROMPT if "elephant" in training_args.run_name else FEM_SYSTEM_PROMPT,
        },
        batched=True)

    ############################
    # Initialize the SFT Trainer
    ############################

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16)
    model.config._name_or_path = model_args.model_name_or_path
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        inference_mode=False,
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

    # Check for last checkpoint
    ckpt = None
    if training_args.resume_from_checkpoint is not None:
        ckpt = training_args.resume_from_checkpoint
        training_args.num_train_epochs += 10
    elif os.path.isdir(training_args.output_dir):
        ckpt = get_last_checkpoint(training_args.output_dir)
        if ckpt:
            logger.info(f"\nCheckpoint detected, resuming training at {ckpt=}.")
            training_args.num_train_epochs += 10
        else:
            logger.info("\nNo checkpoint detected, starting training from scratch.")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[FixedPromptEvaluationCallback(
            model=model,
            tokenizer=tokenizer,
            prompt=FIXED_ELEPHANT_PROMPT if "elephant" in training_args.run_name else FIXED_PROMPT,
            reference=FIXED_ELEPHANT_PROMPT_BASE_MODEL_COMPLETION if "elephant" in training_args.run_name else FIXED_PROMPT_BASE_MODEL_COMPLETION
        )])

    #########################
    # Training and Evaluation
    #########################

    train_result = trainer.train(resume_from_checkpoint=ckpt)
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    trainer.model.save_pretrained(training_args.output_dir + "/final_adapter")

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()