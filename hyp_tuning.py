import optuna
import os
import sys
from optuna.storages import RDBStorage
import argparse
from main import main

username = os.getenv('USER')
INPUT_DIR = os.path.join('/disk/scratch', username,'Cell2Fire', 'data', 'Sub20x20') +'/'
OUTPUT_DIR = os.path.join('/disk/scratch', username,'Cell2Fire', 'results', 'Sub20x20') +'/'

def objective(trial):
    args = {
        'num_epochs': 200,
        'episodes': 3,
        'input_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'lr': trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        'clip_epsilon': trial.suggest_float("clip_epsilon", 0.05, 0.3),
        'value_loss_coef': trial.suggest_float("value_loss_coef", 0.1, 1.0),
        'entropy_coef': trial.suggest_float("entropy_coef", 1e-4, 0.1, log=True),
        'gamma': trial.suggest_float("gamma", 0.95, 0.999),
        'gae_lambda': trial.suggest_float("gae_lambda", 0.9, 0.99),
        'scheduler': trial.suggest_categorical("scheduler", ['cosine', 'step']),
        'T_max': trial.suggest_int('T_max', 5, 50),
        'eta_min': trial.suggest_float('eta_min', 1e-7, 1e-5, log=True)
    }

    return main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db")
    parser.add_argument("--study-name", type=str, default="ppo_optimization")
    args = parser.parse_args()

    # Create storage and study
    storage = RDBStorage(url=args.storage)
    
    # Create study if it doesn't exist
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=False,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage
        )

    # Configure sampler for parallel safe operation
    study.sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=20  # Should be >= number of parallel workers
    )

    study.optimize(objective, n_trials=50)  # Total trials across all workers

    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Average Reward): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")