"""Train causal relation SFT on literal labels and existing entity references.

Uses the shared LoRA trainer and exposure accounting. Current data admission is
recipe-verified complete artificial supervision; medical migration is separate.
"""

from train_choices import main as _main


def main(argv=None):
    return _main(argv, relation_format="labels")


if __name__ == "__main__":
    main()
