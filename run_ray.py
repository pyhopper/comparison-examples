import argparse
import os
import time

from ray import air, tune

from imdb_transformer import train_transformer_imdb
from train_walker2d import train_walker2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune resnet")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--dataset", default="imdb")

    args = parser.parse_args()
    start_time = time.time()

    if args.dataset == "imdb":
        search_space = {
            "ff_dim": tune.qlograndint(64, 512, 64),
            "num_heads": tune.randint(4, 8),
            "dim": tune.qlograndint(16, 128, 16),
            "warumup_steps": tune.qrandint(100, 1000, 100),
            "decay_epochs": tune.qrandint(5, 10, 5),
            "decay_rate": tune.quniform(0.2, 1.0, 0.1),
            "lr": tune.qloguniform(1e-4, 1e-2, 1e-4),
            "wd": tune.qloguniform(1e-6, 1e-4, 1e-6),
            "dr": tune.quniform(0.0, 0.3, 0.1),
            "num_layers": tune.randint(2, 6),
            "embedding_norm": tune.choice([True, False]),
            "embedding_dr": tune.quniform(0.0, 0.3, 0.1),
        }
        tuner = tune.Tuner(
            tune.with_resources(
                train_transformer_imdb,
                {"gpu": 1.0},
                # {"gpu": 0.5},
            ),
            tune_config=tune.TuneConfig(
                num_samples=args.samples,
                mode="max",
            ),
            param_space=search_space,
        )
    elif args.dataset == "walker2d":
        search_space = {
            "units": tune.qrandint(64, 512, 64),
            "warumup_steps": tune.qrandint(100, 500, 100),
            "decay_epochs": tune.choice([5, 10, 20, 50]),
            "decay_rate": tune.quniform(0.2, 1.0, 0.1),
            "lr": tune.qloguniform(1e-4, 1e-2, 1e-4),
            "wd": tune.qloguniform(1e-6, 1e-4, 1e-6),
            "dr": tune.quniform(0.0, 0.3, 0.1),
        }
        tuner = tune.Tuner(
            tune.with_resources(
                train_walker2d,
                {"gpu": 1.0},
            ),
            tune_config=tune.TuneConfig(
                num_samples=args.samples,
                mode="min",
            ),
            param_space=search_space,
        )
    results = tuner.fit()
    print("dataset", args.dataset)
    print("time:", str(time.time() - start_time))
    best = results.get_best_result()
    print("results:", best)
    print("results:", best.metrics)

    basepath_prefix = ""
    if os.path.isdir("/data/pyhopper/"):
        basepath_prefix = "/data/hp_results/"
        os.makedirs(basepath_prefix, exist_ok=True)

    with open(f"{basepath_prefix}ray.txt", "w") as f:
        f.write("best: " + str(best) + "\n")
        f.write("best m: " + str(best.metrics) + "\n")
        f.write("time: " + str(time.time() - start_time) + "\n")