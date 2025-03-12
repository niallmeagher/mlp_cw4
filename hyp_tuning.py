import optuna
import os
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
        'gae_lambda': trial.suggest_float("gae_lambda", 0.9, 0.99)
    }

    return main(args)

if __name__ == "__main__":
    # Create a study to maximize the final reward
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # Adjust number of trials

    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Average Reward): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")