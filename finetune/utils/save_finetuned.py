from transformers import AutoModelForCausalLM
from peft import PeftModel, AutoPeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct", device_map="auto")

print("Loading TOMMI-0.2-10epochs ...")
model = PeftModel.from_pretrained(
    base_model, "/home1/shangsha/workspace/ai-ta/TOMMI-0.2-10epochs-src")
model.save_pretrained("/home1/shangsha/workspace/ai-ta/TOMMI-0.2-10epochs")

del model

print("Loading TOMMI-0.2-10epochs ...")
model = PeftModel.from_pretrained(
    base_model, "/home1/shangsha/workspace/ai-ta/TOMMI-overfit-10epochs-src")
model.save_pretrained("/home1/shangsha/workspace/ai-ta/TOMMI-overfit-10epochs")

del model
