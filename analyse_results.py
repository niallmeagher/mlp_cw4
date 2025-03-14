import optuna
import os
import pandas as pd

# Expand ~ to full path
db_path = os.path.expanduser("~/shared_storage/optuna.db")
storage = f"sqlite:///{db_path}"

# Load study from SQLite
study = optuna.load_study(
    study_name="ppo_20x20",
    storage=storage
)

# Get best trial
best_trial = study.best_trial
print(f"Best trial value (reward): {best_trial.value:.4f}")
print("Best parameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")

# Export to pandas DataFrame
df = study.trials_dataframe()
df.to_csv("optuna_results.csv", index=False)