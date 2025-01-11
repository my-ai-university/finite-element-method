from accelerate import Accelerator
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from finetune.utils.model_utils import (
    get_model_tokenizer,
    check_model_initialization,
    check_model_distribution)
from finetune.utils.gpu_utils import print_all_gpus, check_gpu_usage


if __name__ == "__main__":
    print("Testing Accelerate on multiple GPUs ...")

    print_all_gpus()

    your_hf_token = "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"
    model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "meta-llama/Llama-3.2-11B-Vision"
    # model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    print("Loading model and tokenizer ...")
    model, tokenizer = get_model_tokenizer(
        model_name,
        your_hf_token,
        if_arch_then_weights=False) # default direct loading
    print("Model and tokenizer loaded.")

    print("Initializing accelerator ...")
    accelerator = Accelerator(mixed_precision="bf16", device_placement=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    print("Accelerator initialized.")

    check_gpu_usage()
    check_model_initialization(model)
    check_model_distribution(model)

    # Run a forward pass to verify GPU utilization
    accelerator.print("Running forward pass on multiple GPUs.")
    test_text = "Hello, how are you?" # batch_size = 1
    inputs = tokenizer(test_text,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True).to(accelerator.device)
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Model output shape: {outputs.logits.shape}")
    check_gpu_usage()

    print("Test completed successfully on multiple GPUs.")
