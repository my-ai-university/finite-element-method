from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from huggingface_hub import login, snapshot_download
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


def check_model_distribution(model):
    device_ids = set()
    for name, param in model.named_parameters():
        device_id = param.device
        device_ids.add(device_id)
        print(f"Parameter '{name}' is on device {device_id}")
    print(f"\nModel is distributed across devices: {device_ids}")

def check_model_initialization(model):
    print("\nChecking model initialization:")
    all_good = True  # Flag to track if all parameters are initialized correctly
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Parameter {name} contains NaN or Inf.")
            all_good = False
    if all_good:
        print("All layers are well initialized!")
    else:
        print("Some layers contain invalid values (NaN or Inf). Please check the model initialization.")


def get_model_tokenizer(model_name, token, if_arch_then_weights=False, use_bnb=False):
    login(token=token)

    # Configure BitsAndBytes for 4-bit loading if use_bnb is True
    bnb_config = None
    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Use double quantization for better accuracy
            bnb_4bit_quant_type='nf4',  # Quantization type: 'nf4' or 'fp4'
            bnb_4bit_compute_dtype=torch.float16  # Compute dtype
        )

    if if_arch_then_weights:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config if use_bnb else None,
                device_map=None  # We'll handle device mapping manually
            )

        # Download weights
        weights_location = snapshot_download(
            repo_id=model_name,
            local_dir="/expanse/lustre/scratch/swang31/temp_project/.cache",
            token=token
        )

        # Define maximum memory per GPU
        max_memory = {
            0: "24GiB",  # Limit GPU 0 to 24 GB
            1: "24GiB",  # Limit GPU 1 to 24 GB
        }

        # Infer device map
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"]
        )

        # Load checkpoint with dispatch
        model = load_checkpoint_and_dispatch(
            model=model,
            checkpoint=weights_location,
            device_map=device_map,
            dtype=torch.float16,  # Use float16 for model weights
            quantization_config=bnb_config if use_bnb else None
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=bnb_config if use_bnb else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Reinitialize parameters containing NaNs or Infs
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"\nParameter {name} contains NaN or Inf. Reinitializing...")
            with torch.no_grad():
                param.data.normal_()

    return model, tokenizer
