import argparse
import json
from huggingface_hub import login
import optuna
from optuna_integration.wandb import WeightsAndBiasesCallback
import os
import pexpect
from transformers.trainer_utils import set_seed
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int)
    parser.add_argument('--num_processes', type=int)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--accelerate_config', type=str)
    parser.add_argument('--py_launch_script', type=str)
    parser.add_argument('--wandb_project', type=str)
    args, _ = parser.parse_known_args()
    return args

def objective(trial):
    # define the hyperparameters to optimize: 3 * 3 * 2 * 1 * 3 = 54
    _lora_r = trial.suggest_int("lora_r", 8, 64, log=True) # 16, 32, 64
    _lora_alpha = trial.suggest_int("lora_alpha", 32, 128, log=True) # 32, 64, 128
    _lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.1, step=0.05) # 0.05, 0.1
    _lora_target_modules = trial.suggest_categorical("lora_target_modules", [
        # "q_proj,k_proj,v_proj",  # Attention only
        # "q_proj,k_proj,v_proj,o_proj",  # Full attention
        # "q_proj,k_proj,v_proj,o_proj,gate_proj",  # Attention + gate
        # "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj",  # Attention + FFN
        "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"  # All modules
    ])
    _learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True) # 1e-5, 1e-4, 1e-3

    # build the command to run the training script with DeepSpeed
    _seed = trial.number + 42

    command_str = (
        f"accelerate launch --num_processes {trial.user_attrs['num_processes']} --config_file {trial.user_attrs['accelerate_config']} {trial.user_attrs['py_launch_script']} "
        
        # ** model arguments **
        f"--model_name_or_path 'meta-llama/Llama-3.2-11B-Vision-Instruct' "    
        f"--lora_r {_lora_r} "
        f"--lora_alpha {_lora_alpha} "
        f"--lora_dropout {_lora_dropout} "
        f"--lora_target_modules {_lora_target_modules} "
        
        # ** data arguments **
        f"--data_file {trial.user_attrs['data_file']} "
        f"--split_ratio {0.1} "
        
        # ** training arguments **
        f"--seed {_seed} "
        f"--run_name {trial.user_attrs['run_name']} "
        f"--report_to 'wandb' "
        f"--bf16 {True} "
        f"--dataset_text_field 'text' "
        f"--do_eval {True} "
        f"--num_train_epochs {5} "
        f"--per_device_train_batch_size {18} " # need 4V100 or 2A40-level gpus, decrease to 8 otherwise
        f"--per_device_eval_batch_size {8} "
        f"--gradient_checkpointing {True} "
        f"--gradient_accumulation_steps {2} "
        f"--learning_rate {_learning_rate} "
        f"--max_seq_length {500} "
        f"--warmup_steps {100} "
        f"--weight_decay {0.01} "
        f"--save_strategy 'steps' "
        f"--save_steps {100} "
        f"--save_total_limit {10} "
        f"--logging_strategy 'steps' "
        f"--log_level 'info' "
        f"--logging_steps {1} "
        f"--overwrite_output_dir {True} "
    )

    try:
        print(f"Running trial {trial.number} with command:\n {command_str}\n")
        process = pexpect.spawn(command_str,
                                encoding='utf-8',
                                timeout=36000) # 6000 minutes, just a random large upper bound
        try:
            while True:
                line = process.readline()
                if line == "":
                    break
                print(line, end="")
        except pexpect.EOF:
            print("Process finished.")
        except Exception as e:
            print(f"An error occurred: {e}")
        process.close()

        with open(logging_dir + f"/{trial.user_attrs['run_name']}" + f"/trial_{trail_number}.txt", 'r') as f:
            metrics = json.load(f)
        loss = metrics['loss']

    except:
        loss = float('inf')

    return loss


def main():
    args = parse_args()

    logging_dir = os.environ["LOGGING_DIR"]

    print(f"\nRunning hpo for with {args.n_trials} trials")

    def objective_hpo(trial):
        trial.set_user_attr('num_processes', args.num_processes)
        trial.set_user_attr('data_file', args.data_file)
        trial.set_user_attr('run_name', args.run_name)
        trial.set_user_attr('accelerate_config', args.accelerate_config)
        trial.set_user_attr('py_launch_script', args.py_launch_script)
        trial.set_user_attr('logging_dir', logging_dir)
        return objective(trial)

    study = optuna.create_study(
        study_name="TOMMI-hpo",
        storage=f"sqlite:///{logging_dir}/tommi_hpo.db",
        load_if_exists=True,
        directions=["minimize"])

    study.optimize(
        objective_hpo,
        n_trials=args.n_trials)

    # Print the best parameters
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")


if __name__ == "__main__":
    main()