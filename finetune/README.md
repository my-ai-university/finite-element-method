# AI-TA (Finetune Branch)

This branch is dedicated to developing fine-tuning and related capabilities like hyperparameter optimization (hpo) for the LLM. 

# Quick Tour

You should run every single command under the main `comp-phys-transformer` folder and ensure the files are executable by running the following command 
```bash
find ./ -type f \( -name "*.sh" \) -exec chmod +x {} +
```

## Env Setup

Key env version: `python 3.10`, `cuda 11.8`, `torch 2.5.1`, `deepspeed 0.15.4`, and `bitsandbytes 0.45.0`.

**!!! Try to use local env set up rather than shared project file system on Expanse, activation hanging and missing package can happen :( !!!**

```bash
# set up conda and mamba
./finetune/scripts/set/env/set_conda.sh
source ~/.bashrc
```

If you don't want to use the `mamba` package manager, you can use the `conda` instead by replacing the `mamba` with `conda` in every script. 

```bash
# env set up for fine-tuning and hpo
mamba create -n ai_ta_hpo python=3.10
mamba activate ai_ta_hpo
./finetune/scripts/set/task/set_hpo.sh
```

## Hyperparameter Optimization


### Current HPO Space

Please change your part of the hpo space in the `objective` function in the `./finetune/hpo/main.py` file.

* learning_rate: [1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 6e-4, 7e-4, 8e-4, 1e-3]
* *gradient_accumulation_steps*: [1, 2, 4]
* *epoch*: [3, 5, 10]
* r (lora rank): [8, 16, 32, 64]
* lora_alpha: [16, 32, 64, 128]
* lora_dropout: [0.05, 0.1]
* target_modules
  * "1": ["q_proj", "v_proj"]
  * "2": ["q_proj", "k_proj", "v_proj"]
  * "3": ["q_proj", "k_proj", "v_proj", "o_proj"]
  * "4": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

Current hyperparameters for fine-tuning:

* learning_rate: 1e-3
* gradient_accumulation_steps: 2
* epoch: 5
* r (lora rank): 32
* lora_alpha: 128
* lora_dropout: 0.05
* target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

For either of the following launching methods, modify the `./finetune/scripts/set/env/set_vars.sh` file to set the `WANDB_API_KEY` and `HF_HOME` for wandb and huggingface beforehand.
* For the `WANDB_API_KEY`, you can get it from the [wandb](https://wandb.ai/site/) website.
* For the `HF_HOME`, you can get it from the [huggingface](https://huggingface.co/) website.

### Interactive slurm job

Run the following command to file a request for GPUs.

**!!! Try NOT to use gpu-shared since caching, sync, and more bugs can happen :( !!!**

```bash
# for Expanse: 4 GPUs, 256GB, 48 hours
srun --partition=gpu --account=wis189 --nodes=1 --ntasks-per-node=1 --gpus=4 --cpus-per-task=16 --mem=256G --time=48:00:00 --export=ALL --pty bash -i
```

After you get the GPUs, run the following command to start the hpo process.
```bash
./finetune/scripts/run/run_hpo.sh
```

### Sbatch-submitted slurm job

Change the output path in the `./finetune/scripts/run/hpo.slurm` file to your own path.

```bash
#SBATCH --output=/expanse/lustre/projects/mia346/swang31/projects/ai_ta/output/slurm/%A_%a.out
```

and then directly submit the job to the slurm scheduler by running
```bash
./finetune/scripts/run/sbatch_hpo.sh
```
