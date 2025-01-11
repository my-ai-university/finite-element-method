import optuna
import subprocess
import argparse
import os
import json

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    lora_r = trial.suggest_int('lora_r', 4, 64)
    target_modules = trial.suggest_categorical('target_modules', ['all'])

    # Create a unique identifier for the trial
    trial_id = trial.number

    # Build the command to run the training script with DeepSpeed
    command = [
        'deepspeed',
        '--num_gpus', '2',  # Adjust based on your setup
        './src/test/test_optuna/train.py',  # The training script
        '--learning_rate', str(learning_rate),
        '--lora_r', str(lora_r),
        '--target_modules', target_modules,
        '--trial_id', str(trial_id),
        '--n_splits', '3',  # Number of folds for k-fold CV
        '--model_name', 'gpt2',
        '--dataset_name', 'wikitext',
        '--dataset_config', 'wikitext-2-raw-v1',
        # Add any other necessary arguments
    ]

    # Run the command as a subprocess
    try:
        print(f"Running trial {trial_id} with learning rate {learning_rate} and LoRA r {lora_r}")
        print(f"Command: {' '.join(command)}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Read the evaluation metric from the output file
        with open(f'results_trial_{trial_id}.json', 'r') as f:
            metrics = json.load(f)
        avg_loss = metrics['average_eval_loss']
        return avg_loss
    except subprocess.CalledProcessError as e:
        # Handle exceptions, possibly logging e.stderr
        print(f"Trial {trial_id} failed with error: {e.stderr.decode()}")
        return float('inf')  # Return a high loss to indicate failure

def main():
    print("Starting hyperparameter optimization ...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5, n_jobs=1)  # Run trials sequentially

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (average_eval_loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
