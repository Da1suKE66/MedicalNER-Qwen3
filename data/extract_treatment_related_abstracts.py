#!/usr/bin/env python3
"""Extract clean abstracts from the three article databases into one JSON file.

The output layout is::

    data/treatment_related_articles/articles.json

Each record contains ``source``, ``doi``, ``title``, and ``abstract``. Missing
DOIs are retained as an empty string. Duplicate records within one source keep
the longer abstract; conflicts and extraction totals are stored in the same JSON.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = DATA_DIR / "treatment_related_articles"
SOURCE_DIR_NAMES = {
    "pubmed": "pubmed",
    "Embase": "Embase",
    "cochrane": "cochrane",
}

csv.field_size_limit(100_000_000)


def clean_abstract(value: str, *, remove_abstract_prefix: bool = False) -> str:
    """Normalize export artifacts without rewriting the medical content."""

    value = html.unescape(value or "")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    if remove_abstract_prefix:
        value = re.sub(r"^\s*Abstract\s*[-‐-―:]\s*", "", value, flags=re.I)

    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", value):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs).strip()


def normalize_doi(value: str) -> str:
    value = html.unescape(value or "").strip()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi\s*:\s*", "", value, flags=re.I)
    return value.strip().rstrip(" .;,").casefold()


def normalize_title(value: str) -> str:
    value = html.unescape(value or "")
    value = unicodedata.normalize("NFC", value)
    return re.sub(r"\W+", " ", value.casefold()).strip()


def truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    encoded = encoded[:max_bytes]
    while encoded:
        try:
            return encoded.decode("utf-8").rstrip(" ._")
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return "UNTITLED"


def safe_filename_stem(value: str, *, max_bytes: int = 160) -> str:
    """Make a deterministic Windows-safe filename while retaining readable text."""

    value = html.unescape(value or "")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("/", "__").replace("\\", "__")
    value = re.sub(r'[<>:"|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    value = truncate_utf8(value, max_bytes).strip(" .") or "UNTITLED"
    if value.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        value = "_" + value
    return value


@dataclass
class Article:
    source: str
    title: str
    abstract: str
    source_file: str
    doi: str = ""
    database_id: str = ""
    id_type: str = ""


@dataclass
class ArticleEntry:
    source: str
    doi: str
    database_id: str
    id_type: str
    title: str
    abstract: str
    abstract_chars: int
    selected_source_file: str
    source_files: set[str] = field(default_factory=set)
    duplicate_count: int = 0
    content_conflict_count: int = 0
    longer_replacement_count: int = 0
    alternative_titles: set[str] = field(default_factory=set)
    alternative_identifiers: set[str] = field(default_factory=set)

    def as_record(self) -> dict[str, str]:
        return {
            "source": self.source,
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
        }

    def as_conflict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "doi": self.doi,
            "database_id": self.database_id,
            "id_type": self.id_type,
            "title": self.title,
            "abstract_chars": self.abstract_chars,
            "selected_source_file": self.selected_source_file,
            "source_files": sorted(self.source_files),
            "source_record_count": 1 + self.duplicate_count,
            "duplicate_count": self.duplicate_count,
            "content_conflict_count": self.content_conflict_count,
            "longer_replacement_count": self.longer_replacement_count,
            "alternative_titles": sorted(self.alternative_titles),
            "alternative_identifiers": sorted(self.alternative_identifiers),
        }


class AbstractWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.entries: dict[tuple[str, str], ArticleEntry] = {}
        self.aliases: dict[tuple[str, str], tuple[str, str]] = {}
        self.stats: dict[str, Counter[str]] = {
            source: Counter() for source in SOURCE_DIR_NAMES
        }

    @staticmethod
    def article_aliases(article: Article) -> list[str]:
        aliases: list[str] = []
        if article.doi:
            aliases.append("doi:" + article.doi)
        if article.database_id:
            aliases.append("id:" + article.database_id.casefold())
        title_key = normalize_title(article.title)
        if title_key:
            aliases.append("title:" + title_key)
        return aliases

    @staticmethod
    def proposed_key(article: Article) -> tuple[str, str]:
        if article.doi:
            return "doi:" + article.doi, "doi"
        if article.database_id:
            return "id:" + article.database_id.casefold(), "database_id"
        return "title:" + (normalize_title(article.title) or "untitled"), "title"

    def add(self, article: Article) -> None:
        stats = self.stats[article.source]
        stats["records_seen"] += 1
        article.abstract = clean_abstract(
            article.abstract, remove_abstract_prefix=article.source == "cochrane"
        )
        article.doi = normalize_doi(article.doi)
        article.title = clean_abstract(article.title)
        article.database_id = clean_abstract(article.database_id)

        if not article.abstract:
            stats["skipped_empty_abstract"] += 1
            return

        aliases = self.article_aliases(article)
        key: tuple[str, str] | None = None
        for alias in aliases:
            key = self.aliases.get((article.source, alias))
            if key is not None:
                break

        identity, identity_basis = self.proposed_key(article)
        if key is None:
            key = (article.source, identity)

        entry = self.entries.get(key)

        if entry is None:
            entry = ArticleEntry(
                source=article.source,
                doi=article.doi,
                database_id=article.database_id,
                id_type=article.id_type,
                title=article.title,
                abstract=article.abstract,
                abstract_chars=len(article.abstract),
                selected_source_file=article.source_file,
                source_files={article.source_file},
            )
            self.entries[key] = entry
            stats["unique_records"] += 1
            stats["identity_" + identity_basis] += 1
        else:
            entry.duplicate_count += 1
            entry.source_files.add(article.source_file)
            stats["duplicate_records"] += 1
            if article.title and article.title != entry.title:
                entry.alternative_titles.add(article.title)
            for value in (article.doi, article.database_id):
                if value and value not in {entry.doi, entry.database_id}:
                    entry.alternative_identifiers.add(value)

            if article.abstract != entry.abstract:
                entry.content_conflict_count += 1
                stats["content_conflicts"] += 1
                if len(article.abstract) > entry.abstract_chars:
                    entry.abstract = article.abstract
                    entry.abstract_chars = len(article.abstract)
                    entry.selected_source_file = article.source_file
                    entry.title = article.title or entry.title
                    entry.doi = article.doi or entry.doi
                    entry.database_id = article.database_id or entry.database_id
                    entry.id_type = article.id_type or entry.id_type
                    entry.longer_replacement_count += 1
                    stats["longer_abstract_replacements"] += 1

        for alias in aliases:
            self.aliases[(article.source, alias)] = key

    def write_json(self) -> Path:
        output_path = self.output_root / "articles.json"
        ordered_entries = sorted(
            self.entries.values(),
            key=lambda item: (item.source.casefold(), item.doi, normalize_title(item.title)),
        )

        totals = Counter()
        sources: dict[str, dict[str, int]] = {}
        for source, values in self.stats.items():
            values["output_records"] = sum(
                1 for entry in ordered_entries if entry.source == source
            )
            sources[source] = dict(sorted(values.items()))
            totals.update(values)

        summary = {
            "output_file": str(output_path),
            "deduplication_scope": "within each source database only",
            "conflict_policy": "keep the longer cleaned abstract",
            "missing_doi": "retained as an empty string",
            "sources": sources,
            "totals": dict(sorted(totals.items())),
        }
        conflicts = [entry for entry in ordered_entries if entry.content_conflict_count]

        # Stream the large record array so no second full copy is built in memory.
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write('{\n  "schema_version": "treatment-related-articles-v1",\n')
            handle.write('  "summary": ')
            handle.write(json.dumps(summary, ensure_ascii=False, indent=2).replace("\n", "\n  "))
            handle.write(',\n  "records": [\n')
            for index, entry in enumerate(ordered_entries):
                if index:
                    handle.write(",\n")
                handle.write("    " + json.dumps(entry.as_record(), ensure_ascii=False))
            handle.write('\n  ],\n  "conflicts": [\n')
            for index, entry in enumerate(conflicts):
                if index:
                    handle.write(",\n")
                handle.write("    " + json.dumps(entry.as_conflict(), ensure_ascii=False))
            handle.write("\n  ]\n}\n")
        return output_path


def relative_source_file(path: Path) -> str:
    try:
        return path.relative_to(DATA_DIR.parent).as_posix()
    except ValueError:
        return str(path)


def extract_cochrane(writer: AbstractWriter) -> None:
    source_dir = DATA_DIR / "cochrane"
    for path in sorted(source_dir.glob("*.csv")):
        print(f"[cochrane] {path.name}", flush=True)
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                database_id = (
                    row.get("Cochrane Review ID") or row.get("CENTRAL ID") or ""
                ).strip()
                writer.add(
                    Article(
                        source="cochrane",
                        title=(row.get("Title") or "").strip(),
                        abstract=(row.get("Abstract") or "").strip(),
                        doi=(row.get("DOI") or "").strip(),
                        database_id=database_id,
                        id_type=(
                            "cochrane_review_id"
                            if row.get("Cochrane Review ID")
                            else "central_id" if row.get("CENTRAL ID") else ""
                        ),
                        source_file=relative_source_file(path),
                    )
                )


def iter_embase_records(path: Path) -> Iterator[dict[str, list[str]]]:
    current: dict[str, list[str]] | None = None
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                if current is not None:
                    yield current
                current = None
                continue

            key = row[0].strip().upper()
            values = [value.strip() for value in row[1:] if value.strip()]
            if key == "TITLE":
                if current is not None:
                    yield current
                current = {"TITLE": values}
            elif current is not None and values:
                current.setdefault(key, []).extend(values)

    if current is not None:
        yield current


def first_value(record: dict[str, list[str]], *keys: str) -> str:
    for key in keys:
        values = record.get(key)
        if values:
            return " ".join(values).strip()
    return ""


def extract_embase(writer: AbstractWriter) -> None:
    source_dir = DATA_DIR / "Embase"
    paths = sorted(source_dir.glob("*_complete.csv"))
    complete_set = set(paths)
    paths.extend(path for path in sorted(source_dir.glob("*.csv")) if path not in complete_set)

    for path in paths:
        print(f"[Embase] {path.name}", flush=True)
        for record in iter_embase_records(path):
            abstract = first_value(record, "ABSTRACT", "ORIGINAL (NON-ENGLISH) ABSTRACT")
            writer.add(
                Article(
                    source="Embase",
                    title=first_value(record, "TITLE", "ORIGINAL (NON-ENGLISH) TITLE"),
                    abstract=abstract,
                    doi=first_value(record, "DOI"),
                    database_id=first_value(
                        record,
                        "EMBASE ACCESSION NUMBER",
                        "MEDLINE PMID",
                    ),
                    id_type=(
                        "embase_accession_number"
                        if record.get("EMBASE ACCESSION NUMBER")
                        else "medline_pmid" if record.get("MEDLINE PMID") else ""
                    ),
                    source_file=relative_source_file(path),
                )
            )


PUBMED_METADATA_PREFIXES = (
    "author information:",
    "comment in",
    "comment on",
    "erratum in",
    "erratum for",
    "update in",
    "update of",
    "retraction in",
    "retraction of",
    "expression of concern",
    "conflict of interest",
    "collaborators:",
    "group author",
    "publication types:",
    "mesh terms:",
    "substances:",
    "grant support:",
    "doi:",
    "pmid:",
    "copyright",
    "©",
)


def load_pubmed_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for path in sorted((DATA_DIR / "pubmed").rglob("*.csv")):
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                pmid = (row.get("PMID") or "").strip()
                if not pmid:
                    continue
                current = metadata.setdefault(pmid, {"title": "", "doi": ""})
                title = (row.get("Title") or "").strip()
                doi = normalize_doi(row.get("DOI") or "")
                if title and not current["title"]:
                    current["title"] = title
                if doi:
                    current["doi"] = doi
    return metadata


def iter_pubmed_blocks(path: Path) -> Iterator[str]:
    """Yield PubMed records using the terminal PMID line as the stable boundary."""

    pmid_pattern = re.compile(r"^PMID:\s*\d+")
    lines: list[str] = []
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            lines.append(line)
            if pmid_pattern.match(line):
                yield "".join(lines)
                lines = []


def pubmed_abstract_from_block(block: str, title: str) -> str:
    paragraphs = [
        clean_abstract(part)
        for part in re.split(r"\n\s*\n", block.replace("\r", ""))
        if clean_abstract(part)
    ]
    candidates: list[str] = []
    title_key = normalize_title(title)
    for index, paragraph in enumerate(paragraphs):
        lower = paragraph.casefold().strip()
        if index < 3:
            continue
        if title_key and normalize_title(paragraph) == title_key:
            continue
        if lower.startswith(PUBMED_METADATA_PREFIXES):
            continue
        if lower in {"no abstract available.", "no abstract available"}:
            continue
        if re.fullmatch(r"(?:doi|pmid):.*", paragraph, flags=re.I | re.S):
            continue
        if len(paragraph) < 80:
            continue
        candidates.append(paragraph)
    return max(candidates, key=len, default="")


def extract_pubmed(writer: AbstractWriter) -> None:
    metadata = load_pubmed_metadata()
    writer.stats["pubmed"]["metadata_pmids"] = len(metadata)
    for path in sorted((DATA_DIR / "pubmed").rglob("*.txt")):
        print(f"[pubmed] {path.relative_to(DATA_DIR / 'pubmed')}", flush=True)
        for block in iter_pubmed_blocks(path):
            pmid_match = re.search(r"(?m)^PMID:\s*(\d+)", block)
            if not pmid_match:
                writer.stats["pubmed"]["skipped_missing_pmid"] += 1
                continue
            pmid = pmid_match.group(1)
            meta = metadata.get(pmid, {"title": "", "doi": ""})
            abstract = pubmed_abstract_from_block(block, meta["title"])
            writer.add(
                Article(
                    source="pubmed",
                    title=meta["title"],
                    abstract=abstract,
                    doi=meta["doi"],
                    database_id=pmid,
                    id_type="pmid",
                    source_file=relative_source_file(path),
                )
            )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Delete and rebuild only the configured treatment-related output root.",
    )
    return parser.parse_args(argv)


def prepare_output(output_root: Path, replace_output: bool) -> None:
    output_root = output_root.resolve()
    if output_root == DATA_DIR.resolve() or DATA_DIR.resolve() not in output_root.parents:
        raise SystemExit("output root must be a child of the repository data directory")
    if output_root.exists() and any(output_root.iterdir()):
        if not replace_output:
            raise SystemExit(
                f"output root is not empty: {output_root}; use --replace-output to rebuild it"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    prepare_output(output_root, args.replace_output)
    writer = AbstractWriter(output_root)

    extract_cochrane(writer)
    extract_embase(writer)
    extract_pubmed(writer)
    output_path = writer.write_json()

    print("\nExtraction complete", flush=True)
    print(json.dumps({k: dict(v) for k, v in writer.stats.items()}, indent=2), flush=True)
    print(f"output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
