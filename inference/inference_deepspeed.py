import time
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.utils.inference_utils import Conversation
from finetune.utils.model_utils import (
    get_model_tokenizer,
    check_model_initialization,
    check_model_distribution)
from finetune.utils.gpu_utils import print_all_gpus, check_gpu_usage


# Initialize DeepSpeed
def initialize_deepspeed(model, inference_config):
    model = deepspeed.init_inference(
        model=model,
        config=inference_config,
        # mp_size=torch.cuda.device_count(),
        dtype=torch.float16,
    )
    return model

# Generate a sample text
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.2-11B-Vision"
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Load the model and tokenizer
    print(f"Loading {model_name} model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    print("Model and tokenizer loaded.")

    # DeepSpeed configuration for inference
    inference_config = {
        "replace_with_kernel_inject": True,  # Use kernel injection for better performance
        "tensor_parallel": {"tp_size": 2},  # Number of GPUs (adjust as per your setup)
    }

    # Initialize the model with DeepSpeed
    print("Initializing DeepSpeed ...")
    model = initialize_deepspeed(model, inference_config)
    print("DeepSpeed initialized.")

    check_gpu_usage()

    # Initialize the conversation object
    system_message = 'You are an expert professor who replies in a helpful way.'
    conv = Conversation(
        model,
        tokenizer,
        model.module.device,
        system_message)

    # Run the conversation loop
    print("Starting conversation ...")
    input_text = ""
    while input_text.lower() != "exit":
        input_text = input("Enter your prompt (type 'exit' to quit): ")

        start_time = time.time()
        response = conv.generate(input_text)
        end_time = time.time()

        print(response)
        print(f"Response time: {end_time - start_time:.2f} seconds")
