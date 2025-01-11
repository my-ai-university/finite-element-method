import torch
import os
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


# https://huggingface.co/blog/pytorch-ddp-accelerate-transformers
def main():
    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    # Load tokenizer and model
    model_name = "meta-llama/Llama-3.2-1B"  # Replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=['text']
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=['text']
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Training arguments with DeepSpeed
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        fp16=True,
        deepspeed="./src/finetune/hpo/ds_config.json",  # Path to DeepSpeed config file
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    main()
