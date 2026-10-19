"""Run literal-label relation generation with a separately bound entity stage.

The entity adapter keeps its own existing receipt. Relation adapters must have
been trained for literal label lines; a numeric adapter is rejected, not renamed.
"""

from run_choice_pipeline import main as _main


def main(argv=None):
    return _main(argv, relation_format="labels")


if __name__ == "__main__":
    main()
