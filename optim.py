import argparse
from pathlib import Path

from .mse_tracking import main as mse_training_main
from .mse_tracking_pretrain import main as mse_pretrain_main


w, h = (256, 128)
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="")
parser.add_argument("--resume", default=False, type=bool)
parser.add_argument("--output_dir", default=f"output/tracking_pretrain_{w}_{h}")
parser.add_argument("--text_encoder", default="bert-base-uncased")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--world_size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist_url", default="env://", help="url used to set up distributed training"
)
parser.add_argument("--distributed", default=True, type=bool)
args = parser.parse_args()

config = {
    "train_file": "data/new_logs",
    "image_root": "data/dataset/train",
    "eval_image_root": "data/dataset/test",
    "bert_config": "configs/config_bert.json",
    "image_res": [w, h],
    "vision_width": 768,
    "embed_dim": 256,
    "batch_size": 64,
    "batch_size_test": 32,
    "temp": 0.07,
    "mlm_probability": 0.15,
    "queue_size": 65536,
    "momentum": 0.995,
    "alpha": 0.4,
    "max_words": 16,
    "dropout": 0.1,
    "optimizer": {"opt": "adamW", "lr": 0.0001, "weight_decay": 0.02},
    "schedular": {
        "sched": "cosine",
        "lr": 0.0001,
        "epochs": 1,
        "min_lr": 1e-05,
        "decay_rate": 1,
        "warmup_lr": 1e-05,
        "warmup_epochs": 20,
        "cooldown_epochs": 0,
    },
}

mse_pretrain_main(args, config)

assert Path(args.output_dir).exists(), (
    "Pretraining must be completed before fine-tuning."
)

checkpoint_path = Path(args.output_dir, "checkpoint_00.pth")
assert checkpoint_path.exists(), "Pretraining checkpoint not found."


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=checkpoint_path.as_posix())
parser.add_argument("--resume", default=False, type=bool)
parser.add_argument("--output_dir", default=f"output/tracking_{w}_{h}")
parser.add_argument("--text_encoder", default="bert-base-uncased")
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--world_size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist_url", default="env://", help="url used to set up distributed training"
)
parser.add_argument("--distributed", default=True, type=bool)
args = parser.parse_args()


config = {
    "train_file": "data/new_logs",
    "image_root": "data/dataset/train",
    "eval_image_root": "data/dataset/test",
    "bert_config": "configs/config_bert.json",
    "image_res": [w, h],
    "vision_width": 768,
    "embed_dim": 256,
    "batch_size": 64,
    "batch_size_test": 32,
    "temp": 0.07,
    "mlm_probability": 0.15,
    "queue_size": 65536,
    "momentum": 0.995,
    "alpha": 0.4,
    "max_words": 16,
    "dropout": 0.1,
    "optimizer": {"opt": "adamW", "lr": 1e-05, "weight_decay": 0.02},
    "schedular": {
        "sched": "cosine",
        "lr": 1e-05,
        "epochs": 2,
        "min_lr": 1e-06,
        "decay_rate": 1,
        "warmup_lr": 1e-05,
        "warmup_epochs": 1,
        "cooldown_epochs": 0,
    },
}

mse_training_main(args, config)
