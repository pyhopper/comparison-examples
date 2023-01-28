import argparse
import os

from hyperopt import hp
from hyperopt import fmin, tpe

from imdb_transformer import train_transformer_imdb
from train_walker2d import train_walker2d
import time


def imdb_neg_of(config):
    return -train_transformer_imdb(config)


# https://hyperopt.github.io/hyperopt/#documentation
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetune resnet")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--dataset", default="imdb")
    args = parser.parse_args()
    start_time = time.time()
    if args.dataset == "imdb":
        objective = imdb_neg_of
        space = {
            "ff_dim": hp.choice("ff_dim", [64, 128, 192, 256, 320, 384, 448, 512]),
            "dim": hp.choice("dim", [16, 32, 48, 64, 80, 96, 128]),
            "num_heads": hp.randint("num_heads", 4, 8),
            "warumup_steps": hp.choice(
                "warumup_steps", [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            ),
            "decay_epochs": hp.choice("decay_epochs", [5, 10]),
            "decay_rate": hp.quniform("decay_rate", 0.2, 1.0, 0.1),
            "lr": hp.qloguniform("lr", 1e-4, 1e-2, 1e-4),
            "wd": hp.qloguniform("wd", 1e-6, 1e-4, 1e-6),
            "dr": hp.quniform("dr", 0.0, 0.3, 0.1),
            "num_layers": hp.randint("num_layers", 2, 6),
            "embedding_norm": hp.choice("embedding_norm", [True, False]),
            "embedding_dr": hp.quniform("embedding_dr", 0.0, 0.3, 0.1),
        }
    elif args.dataset == "walker2d":
        objective = train_walker2d
        space = {
            "units": hp.choice("units", [64, 128, 192, 256, 320, 384, 448, 512]),
            "warumup_steps": hp.choice("warumup_steps", [100, 200, 300, 400, 500]),
            "decay_epochs": hp.choice("decay_epochs", [5, 10, 20, 50]),
            "decay_rate": hp.quniform("decay_rate", 0.2, 1.0, 0.1),
            "lr": hp.qloguniform("lr", 1e-4, 1e-2, 1e-4),
            "wd": hp.qloguniform("wd", 1e-6, 1e-4, 1e-6),
        }

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=args.samples)

    print("dataset", args.dataset)
    print("took:", time.time() - start_time)
    print("results:", str(best))
    basepath_prefix = ""
    if os.path.isdir("/data/pyhopper/"):
        basepath_prefix = "/data/hp_results/"
        os.makedirs(basepath_prefix, exist_ok=True)

    with open(f"{basepath_prefix}hyperopt.txt", "w") as f:
        f.write("best: " + str(best) + "\n")
        f.write("time: " + str(time.time() - start_time) + "\n")