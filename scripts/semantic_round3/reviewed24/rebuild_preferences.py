"""Rebuild reviewed DPO candidates from a resident TRAIN mirror; no API or GPU."""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    if not isinstance(value, bytes):
        value = (value if isinstance(value, str) else canonical(value)).encode()
    return hashlib.sha256(value).hexdigest()


def rebuild(recipe, rows):
    by_id = {r["id"]: r for r in rows}
    result = []
    for entry in recipe["selected"]:
        row = by_id[entry["source_row_id"]]
        assert row["split"] == "train" and sha(row) == entry["source_row_sha256"]
        assert sha(row["completion"]) == entry["chosen_sha256"]
        lines = row["completion"].splitlines(keepends=True)
        index = entry["line_index"]
        assert lines[index].endswith("\n")
        fields = lines[index][:-1].split("\t")
        assert len(fields) == 6
        if entry["semantic_change_kind"] == "subtype_direction_reversal":
            assert fields[2] == "subtype_of"
            fields[0], fields[3] = fields[3], fields[0]
            fields[1], fields[4] = fields[4], fields[1]
        else:
            allowed = {"is_core_symptom_of": "argues_against", "argues_against": "is_core_symptom_of"}
            fields[2] = allowed[fields[2]]
        lines[index] = "\t".join(fields) + "\n"
        rejected = "".join(lines)
        assert sha(rejected) == entry["rejected_sha256"]
        assert sum(a != b for a, b in zip(row["completion"].splitlines(), rejected.splitlines())) == 1
        result.append({
            "source_row_id": row["id"], "source_row_sha256": sha(row),
            "rejected": rejected,
            "mutation": {"kind": "reviewed_semantic_contradiction", "line_index": index,
                         "semantic_change_kind": entry["semantic_change_kind"],
                         "explanation": entry["public_support_reason"]},
            "review": {"accepted": True, "source_supported": True,
                       "negative_explicitly_contradicted": True,
                       "reviewer_model": "local_agent_source_review_plus_deepseek-flash",
                       "review_id": entry["candidate_id"], "human_adjudicated": False,
                       "complete_chosen_reviewed": True, "medical_gold": False}
        })
    assert len(result) == recipe["release_count"]
    assert len({r["source_row_id"] for r in result}) == len(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("selection_recipe.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    raw = args.source.read_bytes()
    assert sha(raw) == recipe["resident_train_sha256"], "Resident TRAIN identity mismatch"
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    results = rebuild(recipe, rows)
    blob = "".join(canonical(row) + "\n" for row in results)
    if args.output.exists():
        assert args.output.read_text() == blob, "Refuse overwrite"
    else:
        args.output.write_text(blob)
        args.output.chmod(0o600)
    print(canonical({"rows": len(results), "sha256": sha(blob)}))


if __name__ == "__main__":
    main()
