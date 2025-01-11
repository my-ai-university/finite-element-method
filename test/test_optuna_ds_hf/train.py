import numpy as np
import argparse
import os
import json
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from transformers.trainer_utils import set_seed
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--lora_r', type=int, required=True)
    parser.add_argument('--target_modules', type=str, required=True)
    parser.add_argument('--trial_id', type=int, required=True)
    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
    # Include other arguments as needed
    args, _ = parser.parse_known_args()
    set_seed(args.seed)
    return args

def main():
    args = parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset_name}")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)
    dataset = raw_datasets['train']

    # Load tokenizer and model
    print(f"Loading model {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=None if args.target_modules == 'all' else [args.target_modules]
    )
    model = get_peft_model(model, peft_config)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'],
                         truncation=True,
                         max_length=512,
                         padding=False) # Do not pad here

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=['text']
    )

    # Filter out examples with empty input_ids
    def filter_non_empty(example):
        return len(example['input_ids']) > 0

    tokenized_dataset = tokenized_dataset.filter(filter_non_empty)

    # Set up k-fold cross-validation
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_scores = []
    print(f"Starting trial {args.trial_id}")
    for fold, (train_index, val_index) in enumerate(kf.split(tokenized_dataset)):
        print(f"Starting fold {fold + 1}/{args.n_splits}")
        train_dataset = tokenized_dataset.select(train_index)
        eval_dataset = tokenized_dataset.select(val_index)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/trial_{args.trial_id}_fold_{fold}',
            overwrite_output_dir=True,
            num_train_epochs=1,  # Adjust as needed
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=args.learning_rate,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=50,
            seed=args.seed,
            fp16=True,
            deepspeed="./src/finetune/test_optuna/ds_config.json",
            report_to="none",
            dataloader_num_workers=2,
            disable_tqdm=False,
            local_rank=args.local_rank,  # Required for DeepSpeed
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Set to False for causal language modeling
            )
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        fold_scores.append(metrics['eval_loss'])

        # Reset model for next fold
        # trainer.save_model()  # Save the model if needed
        torch.cuda.empty_cache()

    # Calculate average score across folds
    avg_loss = sum(fold_scores) / len(fold_scores)
    print(f"Trial {args.trial_id} complete. Average eval_loss: {avg_loss}")

    # Write results to a file
    result = {
        'trial_id': args.trial_id,
        'average_eval_loss': avg_loss,
        'fold_scores': fold_scores
    }
    with open(f'results_trial_{args.trial_id}.json', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
