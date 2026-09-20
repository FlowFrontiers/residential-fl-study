"""Shared experiment arguments."""

from .config import ExperimentConfig, HOME_A_PATH, HOME_B_PATH, MODEL_SIZES, RESULTS_DIR, SEEDS


def add_arguments(parser):
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--feature-set", choices=["baseline_16", "alt_16"], default="baseline_16")
    parser.add_argument("--model-size", choices=list(MODEL_SIZES), default="small")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--home-a", default=HOME_A_PATH)
    parser.add_argument("--home-b", default=HOME_B_PATH)


def config_from_args(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    return ExperimentConfig(hidden_dims=list(MODEL_SIZES[args.model_size]),
                            device=args.device, num_threads=args.num_threads,
                            num_rounds=args.rounds, local_epochs=args.local_epochs,
                            batch_size=args.batch_size,
                            equal_weight=getattr(args, "equal_weight", False))
