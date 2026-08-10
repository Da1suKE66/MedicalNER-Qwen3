#!/usr/bin/env python3
"""Extract clean abstracts from the three article databases into one JSON file.

The output layout is::

    data/treatment_related_articles/articles.json

Each record contains ``source``, ``disease_codes``, ``doi``, ``title``, and
``abstract``. Missing DOIs are retained as an empty string. Duplicate records
within one source keep the longer abstract and union all matched disease codes;
conflicts and extraction totals are stored in the same JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import unicodedata
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = DATA_DIR / "treatment_related_articles"
SOURCE_DIR_NAMES = {
    "pubmed": "pubmed",
    "Embase": "Embase",
    "cochrane": "cochrane",
}
SCHEMA_VERSION = "treatment-related-articles-v2"
IDENTITY_POLICY = {
    "pubmed": "PMID",
    "cochrane": "Cochrane Review ID or CENTRAL ID",
    "Embase": (
        "normalized DOI plus normalized title; without DOI, normalized title "
        "plus SHA-256 of the cleaned abstract"
    ),
}

csv.field_size_limit(100_000_000)

T = TypeVar("T")
R = TypeVar("R")
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def ordered_thread_map(
    function: Callable[[T], R], items: Iterable[T], *, workers: int
) -> Iterator[R]:
    """Map files concurrently while yielding results in deterministic order.

    At most ``workers`` parsed file results are retained at once. This keeps
    memory bounded for the large PubMed and Embase exports and preserves the
    existing deterministic duplicate/conflict selection order.
    """

    iterator = iter(items)
    if workers == 1:
        for item in iterator:
            yield function(item)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future[R]] = deque()
        for _ in range(workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break

        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


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
    disease_code: str
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
    disease_codes: set[str] = field(default_factory=set)
    duplicate_count: int = 0
    content_conflict_count: int = 0
    longer_replacement_count: int = 0
    alternative_titles: set[str] = field(default_factory=set)
    alternative_identifiers: set[str] = field(default_factory=set)

    def as_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "disease_codes": sorted(self.disease_codes),
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
        }

    def as_conflict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "disease_codes": sorted(self.disease_codes),
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
        self.pending_empty_disease_codes: dict[tuple[str, str], set[str]] = {}
        self.stats: dict[str, Counter[str]] = {
            source: Counter() for source in SOURCE_DIR_NAMES
        }

    @staticmethod
    def article_identity(article: Article) -> tuple[str, str]:
        """Return a source-aware identity without transitive alias merging.

        PubMed and Cochrane expose database-native identifiers. Embase's compact
        exports do not, and conference/supplement records frequently share one
        DOI, so Embase requires the normalized title as part of the identity.
        Without a DOI, its cleaned abstract digest prevents generic titles such
        as "Early online" from collapsing unrelated records.
        """

        title_key = normalize_title(article.title) or "untitled"
        if article.source in {"pubmed", "cochrane"} and article.database_id:
            return "id:" + article.database_id.casefold(), "database_id"
        if article.source == "Embase":
            if article.doi:
                return f"doi_title:{article.doi}\x1f{title_key}", "doi_title"
            abstract_digest = hashlib.sha256(article.abstract.encode("utf-8")).hexdigest()
            return (
                f"title_abstract:{title_key}\x1f{abstract_digest}",
                "title_abstract",
            )
        if article.doi:
            return "doi:" + article.doi, "doi"
        return "title:" + title_key, "title"

    def add(self, article: Article) -> None:
        stats = self.stats[article.source]
        stats["records_seen"] += 1
        article.abstract = clean_abstract(
            article.abstract, remove_abstract_prefix=article.source == "cochrane"
        )
        article.doi = normalize_doi(article.doi)
        article.title = clean_abstract(article.title)
        article.database_id = clean_abstract(article.database_id)
        article.disease_code = article.disease_code.strip()

        if not article.disease_code:
            raise ValueError(f"missing disease code for {article.source_file}")

        identity, identity_basis = self.article_identity(article)
        key = (article.source, identity)
        entry = self.entries.get(key)

        if not article.abstract:
            stats["skipped_empty_abstract"] += 1
            if entry is not None:
                entry.disease_codes.add(article.disease_code)
            else:
                self.pending_empty_disease_codes.setdefault(key, set()).add(
                    article.disease_code
                )
            return

        if entry is None:
            pending_codes = self.pending_empty_disease_codes.pop(key, set())
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
                disease_codes={article.disease_code, *pending_codes},
            )
            self.entries[key] = entry
            stats["unique_records"] += 1
            stats["identity_" + identity_basis] += 1
        else:
            entry.duplicate_count += 1
            entry.source_files.add(article.source_file)
            entry.disease_codes.add(article.disease_code)
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

    def write_json(self) -> Path:
        output_path = self.output_root / "articles.json"
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        ordered_entries = sorted(
            self.entries.values(),
            key=lambda item: (item.source.casefold(), item.doi, normalize_title(item.title)),
        )

        totals = Counter()
        sources: dict[str, dict[str, int]] = {}
        disease_code_counters: dict[str, Counter[str]] = {
            source: Counter() for source in SOURCE_DIR_NAMES
        }
        for entry in ordered_entries:
            disease_code_counters[entry.source].update(entry.disease_codes)

        for source, values in self.stats.items():
            values["output_records"] = sum(
                1 for entry in ordered_entries if entry.source == source
            )
            sources[source] = dict(sorted(values.items()))
            totals.update(values)

        summary = {
            "output_file": str(output_path),
            "deduplication_scope": "within each source database only",
            "identity_policy": IDENTITY_POLICY,
            "conflict_policy": "keep the longer cleaned abstract",
            "missing_doi": "retained as an empty string",
            "sources": sources,
            "disease_code_counting": (
                "final deduplicated articles; a multi-label article counts once "
                "under each disease code"
            ),
            "disease_code_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in disease_code_counters.items()
            },
            "totals": dict(sorted(totals.items())),
        }
        conflicts = [entry for entry in ordered_entries if entry.content_conflict_count]

        # Stream the large record array so no second full copy is built in memory.
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write('{\n  "schema_version": ')
            handle.write(json.dumps(SCHEMA_VERSION))
            handle.write(',\n')
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
        temporary_path.replace(output_path)
        return output_path


def relative_source_file(path: Path) -> str:
    try:
        return path.relative_to(DATA_DIR.parent).as_posix()
    except ValueError:
        return str(path)


def disease_code_from_path(source: str, path: Path) -> str:
    """Derive the requested disease label from a source export path."""

    if source == "pubmed":
        relative = path.relative_to(DATA_DIR / "pubmed")
        directory = relative.parts[0]
        if re.fullmatch(r"\d{2}(?:-\d{2})?", directory):
            return "F" + directory
    elif source == "cochrane":
        match = re.match(r"^(F\d{2}(?:-\d{2})?)", path.stem)
        if match:
            return match.group(1)
    elif source == "Embase":
        match = re.fullmatch(
            r"F(\d{2})(?:_part\d+)?(?:_complete)?",
            path.stem,
        )
        if match:
            return match.group(1)

    raise ValueError(f"cannot derive {source} disease code from {path}")


def read_cochrane_file(path: Path) -> tuple[Path, list[Article]]:
    articles: list[Article] = []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            database_id = (
                row.get("Cochrane Review ID") or row.get("CENTRAL ID") or ""
            ).strip()
            articles.append(
                Article(
                    source="cochrane",
                    disease_code=disease_code_from_path("cochrane", path),
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
    return path, articles


def extract_cochrane(writer: AbstractWriter, *, workers: int = 1) -> None:
    paths = sorted((DATA_DIR / "cochrane").glob("*.csv"))
    for path, articles in ordered_thread_map(read_cochrane_file, paths, workers=workers):
        print(f"[cochrane] {path.name}", flush=True)
        for article in articles:
            writer.add(article)


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


def read_embase_file(path: Path) -> tuple[Path, list[Article]]:
    articles: list[Article] = []
    for record in iter_embase_records(path):
        abstract = first_value(record, "ABSTRACT", "ORIGINAL (NON-ENGLISH) ABSTRACT")
        articles.append(
            Article(
                source="Embase",
                disease_code=disease_code_from_path("Embase", path),
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
    return path, articles


def extract_embase(
    writer: AbstractWriter, *, workers: int = 1, include_complete: bool = False
) -> None:
    source_dir = DATA_DIR / "Embase"
    normal_paths = sorted(
        path
        for path in source_dir.glob("*.csv")
        if not path.stem.endswith("_complete")
    )
    paths = normal_paths
    if include_complete:
        paths = sorted(source_dir.glob("*_complete.csv")) + normal_paths

    for path, articles in ordered_thread_map(read_embase_file, paths, workers=workers):
        print(f"[Embase] {path.name}", flush=True)
        for article in articles:
            writer.add(article)


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
    "keywords:",
    "doi:",
    "pmid:",
    "copyright",
    "©",
)


def read_pubmed_metadata_file(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            pmid = (row.get("PMID") or "").strip()
            if pmid:
                rows.append(
                    (
                        pmid,
                        (row.get("Title") or "").strip(),
                        normalize_doi(row.get("DOI") or ""),
                    )
                )
    return rows


def load_pubmed_metadata(*, workers: int = 1) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    paths = sorted((DATA_DIR / "pubmed").rglob("*.csv"))
    for rows in ordered_thread_map(read_pubmed_metadata_file, paths, workers=workers):
        for pmid, title, doi in rows:
            current = metadata.setdefault(pmid, {"title": "", "doi": ""})
            if title and not current["title"]:
                current["title"] = title
            if doi:
                current["doi"] = doi
    return metadata


PUBMED_TERMINAL_PMID_PATTERN = re.compile(
    r"^PMID:\s*(\d+)(?:\s+\[[^\]\r\n]+\])?\s*$", re.MULTILINE
)


def iter_pubmed_blocks(path: Path) -> Iterator[str]:
    """Yield PubMed records using the terminal PMID line as the stable boundary."""

    lines: list[str] = []
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            lines.append(line)
            if PUBMED_TERMINAL_PMID_PATTERN.match(line):
                yield "".join(lines)
                lines = []


def pubmed_abstract_from_block(block: str, title: str) -> str:
    paragraphs: list[str] = []
    for part in re.split(r"\n\s*\n", block.replace("\r", "")):
        paragraph = clean_abstract(part)
        if paragraph:
            paragraphs.append(paragraph)
    title_key = normalize_title(title)
    title_index = next(
        (
            index
            for index, paragraph in enumerate(paragraphs)
            if title_key and normalize_title(paragraph) == title_key
        ),
        None,
    )

    # A PubMed text record places its author list immediately after the title.
    # Anchoring on CSV metadata avoids position shifts caused by trailing notes
    # from the preceding record.
    start_index = 0
    if title_index is not None:
        start_index = min(title_index + 2, len(paragraphs))

    abstract_parts: list[str] = []
    for index, paragraph in enumerate(paragraphs[start_index:], start=start_index):
        lower = paragraph.casefold().strip()
        if title_key and normalize_title(paragraph) == title_key:
            continue
        if lower in {"no abstract available.", "no abstract available"}:
            return ""
        if lower.startswith(PUBMED_METADATA_PREFIXES) or re.fullmatch(
            r"(?:doi|pmid):.*", paragraph, flags=re.I | re.S
        ):
            if abstract_parts:
                break
            continue
        # Fallback protection when the title anchor is unavailable: long author
        # lists contain many numbered affiliation markers and are not abstracts.
        if title_index is None and len(
            re.findall(r"\(\d+\)", paragraph[:2000])
        ) >= 5:
            continue
        if title_index is None and index < 3:
            continue
        if len(paragraph) < 80:
            continue
        abstract_parts.append(paragraph)
    return "\n\n".join(abstract_parts)


def read_pubmed_file(
    path: Path, metadata: dict[str, dict[str, str]]
) -> tuple[Path, list[Article], int]:
    articles: list[Article] = []
    skipped_missing_pmid = 0
    for block in iter_pubmed_blocks(path):
        pmid_match = PUBMED_TERMINAL_PMID_PATTERN.search(block)
        if not pmid_match:
            skipped_missing_pmid += 1
            continue
        pmid = pmid_match.group(1)
        meta = metadata.get(pmid, {"title": "", "doi": ""})
        articles.append(
            Article(
                source="pubmed",
                disease_code=disease_code_from_path("pubmed", path),
                title=meta["title"],
                abstract=pubmed_abstract_from_block(block, meta["title"]),
                doi=meta["doi"],
                database_id=pmid,
                id_type="pmid",
                source_file=relative_source_file(path),
            )
        )
    return path, articles, skipped_missing_pmid


def extract_pubmed(writer: AbstractWriter, *, workers: int = 1) -> None:
    metadata = load_pubmed_metadata(workers=workers)
    writer.stats["pubmed"]["metadata_pmids"] = len(metadata)
    paths = sorted((DATA_DIR / "pubmed").rglob("*.txt"))

    def read_path(path: Path) -> tuple[Path, list[Article], int]:
        return read_pubmed_file(path, metadata)

    for path, articles, skipped_missing_pmid in ordered_thread_map(
        read_path, paths, workers=workers
    ):
        print(f"[pubmed] {path.relative_to(DATA_DIR / 'pubmed')}", flush=True)
        writer.stats["pubmed"]["skipped_missing_pmid"] += skipped_missing_pmid
        for article in articles:
            writer.add(article)


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
        help="Atomically replace articles.json while preserving sibling files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent file readers (default: {DEFAULT_WORKERS}; use 1 to disable).",
    )
    parser.add_argument(
        "--include-embase-complete",
        action="store_true",
        help="Also scan redundant *_complete.csv Embase exports (off by default).",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def prepare_output(output_root: Path, replace_output: bool) -> None:
    output_root = output_root.resolve()
    if output_root == DATA_DIR.resolve() or DATA_DIR.resolve() not in output_root.parents:
        raise SystemExit("output root must be a child of the repository data directory")
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "articles.json"
    if output_path.exists() and not replace_output:
        raise SystemExit(
            f"output file already exists: {output_path}; use --replace-output to rebuild it"
        )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    prepare_output(output_root, args.replace_output)
    writer = AbstractWriter(output_root)

    extract_cochrane(writer, workers=args.workers)
    extract_embase(
        writer,
        workers=args.workers,
        include_complete=args.include_embase_complete,
    )
    extract_pubmed(writer, workers=args.workers)
    output_path = writer.write_json()

    print("\nExtraction complete", flush=True)
    print(json.dumps({k: dict(v) for k, v in writer.stats.items()}, indent=2), flush=True)
    print(f"output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
