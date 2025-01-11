import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM


def main(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Simple model and optimizer
    # model = nn.Linear(10, 10).cuda(rank)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").cuda(rank)

    print(f"Rank {rank}: Model is on device {next(model.parameters()).device}")
    ddp_model = DDP(model, device_ids=[rank])

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    world_size = 2
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(main, args=(world_size,), nprocs=world_size)
