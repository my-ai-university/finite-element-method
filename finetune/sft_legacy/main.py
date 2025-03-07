import argparse
import json
from huggingface_hub import login
import optuna
import pexpect
from transformers.trainer_utils import set_seed
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int)
    parser.add_argument('--lr', type=float)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--model_name_or_path', type=str)

    parser.add_argument('--use_4bit_quantization', action='store_true')
    parser.add_argument('--no-use_4bit_quantization', action='store_false')
    parser.set_defaults(use_4bit_quantization=False)

    parser.add_argument('--data_file', type=str)
    parser.add_argument('--accelerate_config', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--logging_dir', type=str)

    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--hf_token', type=str)
    args, _ = parser.parse_known_args()
    return args

def objective(trial):
    # define the hyperparameters
    _r = trial.suggest_categorical("r", [32])
    _lora_alpha = trial.suggest_categorical("lora_alpha", [128])
    _lora_dropout = trial.suggest_categorical("lora_dropout", [0.05])
    _target_modules = trial.suggest_categorical("target_modules", [
        "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"])

    # _num_train_epochs = trial.suggest_categorical("num_train_epochs", [5])
    _num_train_epochs = trial.suggest_categorical("num_train_epochs", [10])
    _gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [2])
    _learning_rate = trial.suggest_categorical("learning_rate", [trial.user_attrs['lr']])

    output_dir = trial.user_attrs['output_dir']
    logging_dir = trial.user_attrs['logging_dir']
    trial_output_file = f'{output_dir}/output_trial_{trial.number}.json'

    # build the command to run the training script with DeepSpeed
    command_str = (
        f"accelerate launch --config_file {trial.user_attrs['accelerate_config']} ./finetune/sft/train.py "
        
        # ** model arguments **
        f"--model_name_or_path {trial.user_attrs['model_name_or_path']} "
        f"--use_4bit_quantization {trial.user_attrs['use_4bit_quantization']} "
        f"--r {_r} "
        f"--lora_alpha {_lora_alpha} "
        f"--lora_dropout {_lora_dropout} "
        f"--target_modules {_target_modules} "
        
        # ** training arguments **
        f"--seed {42} "
        f"--bf16 {True} "
        f"--num_train_epochs {_num_train_epochs} "
        f"--per_device_train_batch_size {18} " # 18 for z3 non-quantized and z2 quantized, 8 for z2 non-quantized 
        f"--per_device_eval_batch_size {8} "
        f"--gradient_checkpointing {True} "
        f"--gradient_accumulation_steps {_gradient_accumulation_steps} "
        f"--learning_rate {_learning_rate} "
        f"--warmup_steps {100} "
        f"--weight_decay {0.01} "
        f"--run_name {trial.user_attrs['run_name']} "
        f"--save_strategy 'epoch' "
        f"--save_total_limit {2} "
        f"--eval_strategy 'no' "
        f"--logging_strategy 'steps' "
        f"--log_level 'info' "
        f"--logging_steps {50} "
        f"--logging_dir {logging_dir} "
        f"--output_dir {output_dir} "
        f"--overwrite_output_dir {True} "
        
        # ** data arguments **
        f"--data_file {trial.user_attrs['data_file']} "
        f"--split_ratio {0.1} "
        f"--max_seq_length {500} "
        
        # ** utils arguments **
        f"--trial_output_file {trial_output_file} "
        f"--trial_number {trial.number} "
    )

    try:
        print(f"Running trial {trial.number} with command:\n {command_str}\n")
        process = pexpect.spawn(command_str,
                                encoding='utf-8',
                                timeout=36000) # 600 minutes, just a random large upper bound
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

        with open(trial_output_file, 'r') as f:
            metrics = json.load(f)
        loss = metrics['loss']

    except:
        loss = float('inf')

    wandb.log({
        "eval_loss": loss,
        "params": trial.params,
    })
    return loss


def main():
    args = parse_args()
    set_seed(args.seed)

    # set up hf and wandb
    login(token=args.hf_token)
    wandb.login(key=args.wandb_api_key)
    if args.wandb_run_id != "":
        print("Resume existing wandb run")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            id=args.wandb_run_id,
            resume="allow",
            config={"optimizer": "Optuna"})
    else:
        print("Start new wandb run")
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={"optimizer": "Optuna"})
        print(f"\nRun ID: {run.id}\n")

    print(f"\nRunning hpo for {args.model_name_or_path} with {args.data_file} for {args.n_trials} trials")
    def objective_finetune_hpo(trial):
        trial.set_user_attr('seed', args.seed)
        trial.set_user_attr('lr', args.lr)
        trial.set_user_attr('model_name_or_path', args.model_name_or_path)
        trial.set_user_attr('use_4bit_quantization', args.use_4bit_quantization)
        trial.set_user_attr('data_file', args.data_file)
        trial.set_user_attr('accelerate_config', args.accelerate_config)
        trial.set_user_attr('run_name', args.run_name)
        trial.set_user_attr('output_dir', args.output_dir)
        trial.set_user_attr('logging_dir', args.logging_dir)
        return objective(trial)

    storage_url = f"sqlite:///{args.output_dir}/hpo_sequential.db"
    study = optuna.create_study(study_name=args.run_name,
                                storage=storage_url,
                                load_if_exists=True,
                                directions=["minimize"])

    def wandb_callback_current_best(study, trial):
        wandb.log({
            "current_best_eval_loss": study.best_trial.value,
            "current_best_params": study.best_trial.params,
        })

    study.optimize(objective_finetune_hpo,
                   n_trials=args.n_trials,
                   callbacks=[wandb_callback_current_best])

    # # optuna visualization
    # figures = {
    #     "optimization_history": optuna.visualization.plot_optimization_history(study),
    #     "slice_plot": optuna.visualization.plot_slice(study),
    #     "parallel_coordinate": optuna.visualization.plot_parallel_coordinate(study),
    #     "contour_plot": optuna.visualization.plot_contour(study),
    #     "param_distribution": optuna.visualization.plot_param_importances(study)
    # }
    #
    # for name, fig in figures.items():
    #     file_path = f"{args.output_dir}/{name}.html"
    #     fig.write_html(file_path)
    #     print(f"Saved {name} to {file_path}")

    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best loss: {best_trial.value}")
    print(f"Best params: {best_trial.params}")

    # wandb logging
    wandb.log({
        "best_trial_eval_loss": best_trial.value,
        "best_trial_params": best_trial.params,
    })

    df = study.trials_dataframe()
    wandb.Table(dataframe=df)
    wandb.finish()


if __name__ == "__main__":
    main()