import time
import torch
import csv
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
from peft import PeftModel
from finetune.utils.model_utils import check_model_distribution
from finetune.utils.gpu_utils import check_gpu_usage
from finetune.utils.prompt import FEM_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, FIXED_ELEPHANT_PROMPT

def main():
    #MODIFY THE PATHS BELOW
    filename = "tommi-0.3-250311"
    output_path = "/project/garikipa_1359/rahulgul/ai-ta/finite-element-method"
    output_csv_path = f"{output_path}/{filename}.csv"
    #questions_csv_path = "/project/garikipa_1359/projects/ai_ta/outputs/250225_train_dataset_2023WN.csv"
    questions_csv_path = "/project/garikipa_1359/rahulgul/ai-ta/finite-element-method/finetune/evaluate/check_dataset1.csv"
    base_model_name_or_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    adapter_model_name_or_path = "/project/garikipa_1359/rahulgul/ai-ta/model2/tommi-0.2"
    adapter_version = "TOMMI-0.3"
    
    print(f"Loading base model: {base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    finetune_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    print(f"Loading fine-tuned model from: {adapter_model_name_or_path}")
    model = PeftModel.from_pretrained(finetune_model, adapter_model_name_or_path)
    model = model.merge_and_unload()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model_name_or_path, return_tensors="pt")
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
            writer.writerow(headers + ["Base Model Answer", "Fine Tuned Model Answer"])
        
        for row in reader:
            if not row or len(row) < 3:
                continue
            ID, question, original_answer = row[:3]
            print(f"Processing question ID: {ID}")
            
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT if "elephant" in adapter_version else FEM_SYSTEM_PROMPT},
                {"role": "user", "content": FIXED_ELEPHANT_PROMPT if "elephant" in adapter_version else question},
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(input_text, max_length=500, truncation=True, return_tensors="pt").to(model.device)
            
            def generate_response(model):
                try:
                    with torch.no_grad():
                        output = model.generate(**inputs, max_length=2000, temperature=0.01, top_k=1, top_p=1.0)
                    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    if "assistant" in response_text:
                        response_text = response_text.split("assistant", 1)[-1].strip()
                    return response_text
                    #return tokenizer.decode(output[0], skip_special_tokens=True)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    return "Error in model response"
            
            base_model_response = generate_response(base_model)
            finetuned_model_response = generate_response(model)
            
            writer.writerow(row + [base_model_response, finetuned_model_response])
            time.sleep(0.1)
    
    print("Processing completed. Results saved to", output_csv_path)

if __name__ == '__main__':
    main()
