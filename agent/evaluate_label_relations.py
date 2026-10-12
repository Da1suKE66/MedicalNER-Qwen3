"""Literal relation-label entry point for the frozen artificial holdout."""

from evaluate_choice_relations import main as evaluate_main


def main(argv=None):
    return evaluate_main(argv, relation_format="labels")


if __name__ == "__main__":
    main()
