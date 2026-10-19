"""Exploratory orthography-only scoring; NEVER replaces the strict primary score.

Created after inspection of historical dev failures. It is not an independent
evaluation or a clinical synonym resolver. Type, word order, subtype and all
relation/evidence labels remain unchanged. No substring/fuzzy concept matching.
"""

import argparse
import copy
import json
from pathlib import Path
import re

from build_stage_data import _message
from evaluate import _graph
from evaluate_graphs import evaluate_records
from kg_agent.normalize import normalize_text


def normalize_name(name):
    words = re.findall(r"\w+", normalize_text(name))
    spelling = {
        "generalized": "generalised",
        "behavior": "behaviour",
        "behavioral": "behavioural",
    }
    return " ".join(spelling.get(word, word) for word in words)


def graph(value):
    result = _graph(value)
    if result is None:
        return None
    for node in result["entities"]:
        node["name"] = normalize_name(
            node.get("name") or node.get("properties", {}).get("Name", "")
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    records = json.loads(Path(args.gold).read_text())
    predictions = [
        json.loads(l)
        for l in Path(args.predictions).read_text().splitlines()
        if l.strip()
    ]
    strict = evaluate_records(records, predictions)
    targets = [
        {
            "messages": [
                {"role": "user", "content": _message(r, "user")},
                {
                    "role": "assistant",
                    "content": json.dumps(graph(_message(r, "assistant"))),
                },
            ]
        }
        for r in records
    ]
    normalized = [{"output": graph(copy.deepcopy(r))} for r in predictions]
    result = {
        "scope": __doc__,
        "strict": strict,
        "orthography_only": evaluate_records(targets, normalized),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]["free_text"]["relation"]
                for k in ["strict", "orthography_only"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
