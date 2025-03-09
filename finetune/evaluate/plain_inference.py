from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, MllamaForCausalLM, MllamaForConditionalGeneration, PreTrainedTokenizerFast

from finetune.utils.eval_utils import get_latest_checkpoint
from finetune.utils.prompt import FEM_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, FIXED_PROMPT, FIXED_ELEPHANT_PROMPT


if __name__ == "__main__":
    base_model_name_or_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    adapter_model_dir = "/project/neiswang_1391/shangsha/ai-ta/finite-element-method/outputs"
    adapter_version = "TOMMI-elephant-auto"

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

    # Prepare input
    messages =     [
        {
            "role": "system", "content": DEFAULT_SYSTEM_PROMPT if "elephant" in adapter_version else FEM_SYSTEM_PROMPT
        },
        {
            "role": "user", "content": FIXED_ELEPHANT_PROMPT if "elephant" in adapter_version else FIXED_PROMPT
        },
    ]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_model_name_or_path,
        # padding_side="left",
        return_tensors="pt")
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = "<|reserved_special_token_5|>"
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False)
    inputs = tokenizer(
        input_text,
        max_length=500,
        truncation=True,
        return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=256,
            temperature=0.01,
            top_k=1,
            top_p=1.0)

    print("Prompt:\n" + input_text)
    print("Completion:\n" + tokenizer.decode(output[0], skip_special_tokens=True))