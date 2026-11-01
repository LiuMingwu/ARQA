import os
import pdb
os.environ["WANDB_DISABLED"] = "true"

from llmtuner import run_exp


def main():
    #pdb.set_trace()
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
