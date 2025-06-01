# from accelerate import Accelerator
import time
import torch
import sys
import os
import csv
import time
from transformers import PreTrainedTokenizerFast
from peft import PeftModel
from transformers import AutoModelForCausalLM
# from huggingface_hub import snapshot_download, login


# from finetune.utils.inference_utils import Conversation
from finetune.utils.model_utils import (
    get_model_tokenizer,
    check_model_initialization,
    check_model_distribution)
from finetune.utils.eval_utils import get_latest_checkpoint
from finetune.utils.gpu_utils import print_all_gpus, check_gpu_usage
from finetune.utils.prompt import FEM_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, FIXED_PROMPT, FIXED_ELEPHANT_PROMPT


def main():
    filename = "tommi-0.2-250310"
    output_path = f"/project/garikipa_1359/projects/ai_ta/outputs"
    output_csv_path = f"{output_path}/{filename}.csv"
    questions_csv_path = f"{output_path}/250225_train_dataset_2023WN.csv"

    base_model_name_or_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # peft_model_name = "my-ai-university/TOMMI-0.2"
    adapter_model_dir = output_path
    adapter_version = "TOMMI-v0.2"

    # Example usage
    adapter_model_name_or_path = get_latest_checkpoint(adapter_model_dir + f"/{adapter_version}")
    print(f"Latest checkpoint: {adapter_model_name_or_path}")

    # Load the model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto") # Automatically distributes across available GPUs
    model = PeftModel.from_pretrained(base_model, adapter_model_name_or_path)
    model = model.merge_and_unload() # Optional: Merge adapter with base model for faster inference

    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_model_name_or_path,
        # padding_side="left",
        return_tensors="pt")
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = "<|reserved_special_token_5|>"

    check_model_distribution(model)
    check_gpu_usage()

    print("Reading input questions from CSV ...")
    with open(questions_csv_path, mode='r', encoding='utf-8') as input_file, \
            open(output_csv_path, mode='w', encoding='utf-8', newline='') as output_file:

        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        headers = next(reader, None)
        if headers:
            writer.writerow(["Fine Tuned Model Response"])

        for row in reader:
            # if not row or len(row) < 3:
            #     continue
            ID = row[0]
            question = row[1]
            answer = row[2]
            print(f"Processing question: {ID}")

            messages =     [
               {
                    "role": "system", "content": DEFAULT_SYSTEM_PROMPT if "elephant" in adapter_version else FEM_SYSTEM_PROMPT
                },
                {
                    "role": "user", "content": FIXED_ELEPHANT_PROMPT if "elephant" in adapter_version else question
                },
            ]

            input_text = tokenizer.apply_chat_template(
                messages,   
                add_generation_prompt=True,
                tokenize=False)
            inputs = tokenizer(
                input_text,
                max_length=500,
                truncation=True,
                return_tensors="pt").to(model.device)

            try:
                with torch.no_grad():
                    response = model.generate(
                        **inputs,
                        max_length=256,
                        temperature=0.01,
                        top_k=1,
                        top_p=1.0)
                response_text = response["content"] if isinstance(response, dict) else response
            except Exception as e:
                print(f"Error processing question: {question}, Error: {e}")
                response_text = "Error in model response"

            writer.writerow([response_text])
            time.sleep(0.1)

    print("Processing completed. Results saved to", output_csv_path)

if __name__ == '__main__':
    main()