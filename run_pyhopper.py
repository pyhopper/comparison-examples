import argparse
import os
import time

import pyhopper
from imdb_transformer import train_transformer_imdb
from train_walker2d import train_walker2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune resnet")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--dataset", default="imdb")
    args = parser.parse_args()
    start_time = time.time()
    if args.dataset == "imdb":
        direction = "max"
        of = train_transformer_imdb
        search_space = {
            "ff_dim": pyhopper.int(64, 512, multiple_of=64),
            "num_heads": pyhopper.int(4, 8),
            "dim": pyhopper.int(16, 128, multiple_of=16),
            "warumup_steps": pyhopper.int(100, 1000, multiple_of=100),
            "decay_epochs": pyhopper.choice([5, 10]),
            "decay_rate": pyhopper.float(0.2, 1.0, "0.1f"),
            "lr": pyhopper.float(1e-4, 1e-2, "0.1g"),
            "wd": pyhopper.float(1e-6, 1e-4, "0.1g"),
            "dr": pyhopper.float(0.0, 0.3, "0.1f"),
            "num_layers": pyhopper.int(2, 6),
            "embedding_norm": pyhopper.choice([True, False]),
            "embedding_dr": pyhopper.float(0.0, 0.3, "0.1f"),
        }
    elif args.dataset == "walker2d":
        direction = "min"
        of = train_walker2d
        search_space = {
            "units": pyhopper.int(64, 512, multiple_of=64),
            "warumup_steps": pyhopper.int(100, 500, multiple_of=100),
            "decay_epochs": pyhopper.choice([5, 10, 20, 50]),
            "decay_rate": pyhopper.float(0.2, 1.0, "0.1f"),
            "lr": pyhopper.float(1e-4, 1e-2, "0.1g"),
            "wd": pyhopper.float(1e-6, 1e-4, "0.1g"),
            "dr": pyhopper.float(0.0, 0.3, "0.1f"),
        }

    search = pyhopper.Search(search_space)
    results = search.run(of, direction, steps=args.samples, n_jobs="per-gpu")

    print("dataset:", args.dataset)
    print("results:", results)
    print("best_f:", search.best_f)
    print("time:", str(time.time() - start_time))
    basepath_prefix = ""
    if os.path.isdir("/data/pyhopper/"):
        basepath_prefix = "/data/hp_results/"
        os.makedirs(basepath_prefix, exist_ok=True)

    with open(f"{basepath_prefix}pyhopper.txt", "w") as f:
        f.write("best: " + str(results) + "\n")
        f.write("best_f: " + str(search.best_f) + "\n")
        f.write("time: " + str(time.time() - start_time) + "\n")