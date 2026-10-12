"""Literal relation-label entry point for the frozen medical oracle subset."""

from evaluate_choice_medical import main as evaluate_main


def main(argv=None):
    return evaluate_main(argv, relation_format="labels")


if __name__ == "__main__":
    main()
