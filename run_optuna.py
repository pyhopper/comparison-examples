import argparse
import os
import time

import optuna

from imdb_transformer import train_transformer_imdb
from train_walker2d import train_walker2d


# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html#sphx-glr-tutorial-10-key-features-001-first-py
# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py


def objective_imdb(trial):
    config = {
        "ff_dim": trial.suggest_int("ff_dim", 64, 512, log=True),
        "num_heads": trial.suggest_int("num_heads", 4, 8),
        "dim": trial.suggest_int("dim", 16, 128, log=True),
        "warumup_steps": trial.suggest_int("warumup_steps", 100, 1000, step=100),
        "decay_epochs": trial.suggest_int("decay_epochs", 5, 10, step=5),
        "decay_rate": trial.suggest_float("decay_rate", 0.2, 1.0, step=0.1),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-4, log=True),
        "dr": trial.suggest_float("dr", 0.0, 0.3, step=0.1),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "embedding_norm": trial.suggest_categorical("embedding_norm", [True, False]),
        "embedding_dr": trial.suggest_float("embedding_dr", 0.0, 0.3, step=0.1),
    }
    return train_transformer_imdb(config, verbose=False)


def objective_walker2d(trial):
    config = {
        "units": trial.suggest_int("units", 64, 512, log=True),
        "warumup_steps": trial.suggest_int("warumup_steps", 100, 500, step=100),
        "decay_epochs": trial.suggest_categorical("decay_epochs", [5, 10, 20, 50]),
        "decay_rate": trial.suggest_float("decay_rate", 0.2, 1.0, step=0.1),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-4, log=True),
    }
    return train_walker2d(config, verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune resnet")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--dataset", default="imdb")
    args = parser.parse_args()

    start_time = time.time()

    if args.dataset == "imdb":
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_imdb, n_trials=args.samples)
    elif args.dataset == "walker2d":
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_walker2d, n_trials=args.samples)

    print("dataset", args.dataset)
    print("time:", str(time.time() - start_time))
    best_params = study.best_params
    best_value = study.best_value

    print("results:", best_params)
    print("best_f:", best_value)
    basepath_prefix = ""
    if os.path.isdir("/data/pyhopper/"):
        basepath_prefix = "/data/hp_results/"
        os.makedirs(basepath_prefix, exist_ok=True)

    with open(f"{basepath_prefix}optuna.txt", "w") as f:
        f.write("best: " + str(best_params) + "\n")
        f.write("best_f: " + str(best_value) + "\n")
        f.write("time: " + str(time.time() - start_time) + "\n")