import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch", "--batch", help="Batch", required=False, default=4, type=int)
    parser.add_argument("-epochs", "--epochs", help="Training Epochs", required=False, default=3)
    parser.add_argument("-save", "--save", help="Save Checkpoints", required=False, default="dialogpt-finetune")
    parser.add_argument("-lr", "--lr", help="Learning Rate", required=False, default=1e-5)
    parser.add_argument("-clip", "--clip", help="Gradient Clip", required=False, default=2.0)
    parser.add_argument("-seed", "--seed", help="Seed", required=False, default=0)
    parser.add_argument("-context", "--context", help="Number Context", required=False, default=3)
    parser.add_argument("-max_len", "--max_len", help="Maximum length of tokens", required=False, default=128, type=int)
    parser.add_argument("-prepare", "--prepare", help="Prepare Dataset", required=False, default=False)
    parser.add_argument(
        "-grad_accumulate", "--grad_accumulate", help="Gradient Accumulation", required=False, default=8, type=int
    )
    parser.add_argument("-tensorboard", "--tensorboard", help="Tensorboard runs", required=False, default="runs/")
    parser.add_argument("-early_stop", "--early_stop", help="Early Stopping", required=False, default=10, type=int)
    parser.add_argument("-eval", "--eval", help="Evaluate", required=False, default=False, type=bool)

    args = vars(parser.parse_args())

    return args
