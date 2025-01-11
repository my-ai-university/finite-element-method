import torch


def print_all_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"{num_gpus} GPUs are being used:")
        for i in range(num_gpus):
            print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Using device: {torch.cuda.get_device_name(0)}")

def clear_gpu_cache(rank=None):
    """
    Clears the GPU cache.

    Args:
        rank (int, optional): The rank of the process. If rank is 0, a log message is printed.
    """
    if rank == 0:
        print("Clearing GPU cache for all ranks.")
    torch.cuda.empty_cache()

def check_gpu_usage():
    print("\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3  # GB
        print(f" - GPU {i}: Allocated Memory: {allocated:.2f} GB, Reserved Memory: {reserved:.2f} GB")
