import optuna


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Return a tuple of objectives: maximize accuracy and minimize training time
    return 0, 0

# Create a multi-objective study
study = optuna.create_study(
    study_name="ai_ta_finetune_hpo_multi",
    storage="sqlite:///ai_ta_finetune_hpo_multi.db",  # Optional: Persist the study
    load_if_exists=True,  # Optional: Load existing study if it exists
    directions=["maximize",
                "minimize"]  # Specify direction for each objective
)

# Optimize the study
study.optimize(objective,
               n_trials=100,
               timeout=600)  # e.g., 100 trials or 10 minutes

# Access Pareto optimal trials
print("Pareto Front:")
for trial in study.best_trials:
    print(f"Accuracy: {trial.values[0]}, Training Time: {trial.values[1]}")
    print(f"Hyperparameters: {trial.params}")
