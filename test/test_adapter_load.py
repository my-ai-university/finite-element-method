import torch
from transformers import MllamaForConditionalGeneration
from peft import PeftModel, PeftConfig


if __name__ == "__main__":
    # Base model ID and LoRA model ID
    base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    lora_model_id = "Kadins/Llama-3.2-Vision-chinese-lora"

    # Load the base model
    base_model = MllamaForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    ).eval()

    # Make sure the adapter is properly loaded
    try:
        # Method 1: Using PeftModel.from_pretrained
        model = PeftModel.from_pretrained(base_model, lora_model_id)

        # Check if PEFT config is loaded
        if hasattr(model, "_hf_peft_config_loaded") and model._hf_peft_config_loaded:
            # Get active adapter name
            adapter_name = model.active_adapter
            print(f"Active adapter: {adapter_name}")

            # Get the adapter state dict
            adapter_state_dict = model.get_adapter_state_dict(adapter_name)

            # Print all keys in the adapter
            print("\nAdapter keys:")
            for key in adapter_state_dict.keys():
                print(f"- {key}")

            # Optionally print shapes
            print("\nAdapter parameters with shapes:")
            for key, value in adapter_state_dict.items():
                print(f"- {key}: {value.shape}")
        else:
            print("No PEFT config loaded. Using alternative method.")

            # Method 2: Using load_adapter method
            base_model.load_adapter(lora_model_id)
            adapter_name = base_model.active_adapters()
            print(f"Active adapter: {adapter_name}")

            adapter_state_dict = base_model.get_adapter_state_dict()

            print("\nAdapter keys:")
            for key in adapter_state_dict.keys():
                print(f"- {key}")

    except ValueError as e:
        print(f"Error loading adapter: {e}")

        # Method 3: Check model status (for debugging)
        if hasattr(model, "get_model_status"):
            print("\nModel status:")
            print(model.get_model_status())

        if hasattr(model, "get_layer_status"):
            print("\nSample layer status:")
            layer_status = model.get_layer_status()
            if layer_status:
                print(layer_status[0])