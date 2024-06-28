"""
This sample script demonstrates how to use a Hugging Face transformer model for various tasks.
The script handles loading the pre-trained model and tokenizer, setting up the device for computation (CPU or GPU),
and generating the text output based on the specified prompt type.

The script supports various prompt types such as summarization, question-answering, and text completion
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set your Hugging Face token explicitly
token = "GET_FROM_HuggingFace_Profile_setting"

# model_name = "mistralai/Mistral-7B-v0.1"
model_name = "meta-llama/Meta-Llama-3-8B"

# Prompt type selection
prompt_type = "summarize"  # see other options below

# length of response
max_new_tokens = 100

# ---------------------------------------------------------------
# Load the tokenizer with authentication, using the slow tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, cache_dir="./hf_cache/")

# Load the model with authentication
model = AutoModelForCausalLM.from_pretrained(model_name, token=token, cache_dir="./hf_cache/")

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using ', device)
model.to(device)


if prompt_type == "qa":
    question = "What is finite element method?"
    input_text = f"Question: {question}\n\nAnswer:"

elif prompt_type == "context_qa":
    context = """
    The finite element method (FEM) is a numerical technique for solving partial 
    differential equations (PDEs) in their weak forms. It's widely used in 
    engineering and scientific computing.
    """
    question = "What should be the form of equations for FEM?"
    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

elif prompt_type == "completion":
    input_text = "Complete the following: The finite element method is used for"

elif prompt_type == "summarize":
    long_text = """
    The finite element method (FEM) is a numerical method for solving problems of 
    engineering and mathematical physics. Typical problem areas of interest include 
    structural analysis, heat transfer, fluid flow, mass transport, and electromagnetic 
    potential. The analytical solution of these problems generally require the solution 
    to boundary value problems for partial differential equations. The finite element 
    method formulation of the problem results in a system of algebraic equations. The 
    method yields approximate values of the unknowns at discrete number of points over 
    the domain. To solve the problem, it subdivides a large problem into smaller, simpler 
    parts that are called finite elements.
    """
    input_text = f"Summarize the following text about FEM:\n{long_text}\n\nSummary:"

elif prompt_type == "compare":
    input_text = "Compare and contrast the Finite Element Method (FEM) with the Finite Difference Method (FDM):"

elif prompt_type == "explain":
    input_text = "Explain the concept of mesh generation in Finite Element Method to a beginner:"

else:
    input_text = "Tell me about the Finite Element Method."


inputs = tokenizer(input_text, return_tensors='pt').to(device)  # pt stands for Pytorch

# Measure inference time
start_time = time.time()

with torch.no_grad():             # Inference, no training back prop. calculations  needed
    outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,           # Maximum number of tokens to generate
    do_sample=True,               # Use sampling instead of greedy decoding
    temperature=0.7,              # Controls randomness in sampling (lower = more deterministic)
    top_k=50,                     # Limits sampling to top k most likely tokens
    top_p=0.95,                   # Nucleus sampling: consider tokens with cumulative probability of 95%
    # num_return_sequences=3,       # Number of alternative sequences to generate
    # no_repeat_ngram_size=2,       # Prevent repetition of n-grams of this size
    # early_stopping=True,          # Stop when all beam hypotheses reached the EOS token
    # length_penalty=1.0,           # Exponential penalty to the length (1.0 means no penalty)
    # repetition_penalty=1.2,       # Penalty for repeating tokens
    # bad_words_ids=None,           # List of token IDs that are not allowed to be generated
    # forced_bos_token_id=None,     # Token ID to force as the first generated token
    # forced_eos_token_id=None,     # Token ID to force as the last generated token
    # num_beam_groups=1,            # Number of groups for diverse beam search
    # diversity_penalty=0.0,        # Penalty for diversity between beam groups
    # encoder_no_repeat_ngram_size=2  # Size of n-grams to prevent repetition in the encoder
)
end_time = time.time()

# Decode the output
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Prompt: {input_text}")
print(f"Answer: {answer}")
print(f"Inference Time: {end_time - start_time:.2f} seconds")