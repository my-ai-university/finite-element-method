from accelerate import Accelerator
import time
import torch

from finetune.utils.inference_utils import Conversation
from finetune.utils.model_utils import (
    get_model_tokenizer,
    check_model_initialization,
    check_model_distribution)
from finetune.utils.gpu_utils import print_all_gpus, check_gpu_usage


if __name__ == "__main__":
    hf_token = "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"
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-3.2-11B-Vision"
    # model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # model_name = "meta-llama/Llama-3.2-90B-Vision-Instruct"

    print(f"Loading {model_name} model and tokenizer ...")
    model, tokenizer = get_model_tokenizer(model_name, hf_token, use_bnb=True)
    print("Model and tokenizer loaded.")

    print("Initializing accelerator ...")
    accelerator = Accelerator(mixed_precision="bf16", device_placement=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)
    print("Accelerator initialized.")

    check_model_distribution(model)
    check_gpu_usage()

    # Initialize the conversation object
    system_message = 'You are an expert professor who replies in a helpful way.'
    conv = Conversation(
        model,
        tokenizer,
        accelerator.device,
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

    # # Save the conversation to a file
    # with open("./conversation.txt", "w") as f:
    #     f.write(str(conv.message))
