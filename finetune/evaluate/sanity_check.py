from accelerate import Accelerator
from datasets import load_dataset
import math
from peft import AutoPeftModelForCausalLM
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed


def evaluate_model(
        model_name,
        datasets,
        max_length=500, # align with the fine-tuning setup
        batch_size=8 # align with the fine-tuning setup
    ):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    if model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    accelerator = Accelerator(mixed_precision="bf16", device_placement=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)

    perplexities = {}
    for key, dataset in datasets.items():
        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating Perplexity"):
            batch_text = dataset[i:i+batch_size]['text']
            encodings = tokenizer(batch_text,
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=max_length,
                                  padding=True)
            input_ids = encodings.input_ids.to(accelerator.device)
            attention_mask = encodings.attention_mask.to(accelerator.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        perplexities[key] = perplexity

    return perplexities


if __name__ == "__main__":
    set_seed(42)

    # Fine-tuned and base models
    base_model = "meta-llama/Llama-3.2-11B-Vision-Instruct" # 12.726826616613353
    fine_tuned_models = [
        "ai-teaching-assistant/TOMMI-0.1", # 12.726826616613353
    ]

    # For faster evaluation => use 1000 lines, 15 mins per model
    datasets = {
        "wikipedia": load_dataset('wikipedia', '20220301.en', split='train[:10000]', trust_remote_code=True)
    }

    # Calculate Perplexity
    with open("./finetune/evaluate/sanity_check_perplexities.txt", "w") as f:
        for model_name in [base_model] + fine_tuned_models:
            torch.cuda.empty_cache()

            print(f"Evaluating model: {model_name}")
            perplexities = evaluate_model(model_name, datasets)
            for dataset, perplexity in perplexities.items():
                print(f"Perplexity for {model_name} on {dataset}: {perplexity}")
                f.write(f"\nPerplexity for {model_name}:\n")
                f.write(f"On {dataset}, {perplexity}\n")