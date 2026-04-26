#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from datetime import datetime
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml

DIM = 384
MAX_K = 100


@dataclass
class Result:
    doc_id: int
    score: float


class LiteralString(str):
    pass


def literal_string_representer(dumper: yaml.Dumper, data: LiteralString) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


yaml.SafeDumper.add_representer(LiteralString, literal_string_representer)


def vlog(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, file=sys.stderr)


def has_path_separator(s: str) -> bool:
    return "/" in s


def build_db_paths(base: str, user_cwd: str) -> tuple[Path, Path]:
    if has_path_separator(base):
        if base.startswith("/"):
            prefix = Path(base)
        else:
            prefix = Path(user_cwd) / base
    else:
        prefix = Path(user_cwd) / base
    return (
        prefix.with_suffix(".memo"),
        prefix.with_suffix(".yaml"),
    )


def ensure_parent_dir(file_path: Path) -> None:
    parent = file_path.parent
    parent.mkdir(parents=True, exist_ok=True)


def load_yaml_tables(path: Path) -> tuple[list[str], list[dict[str, Any] | None]]:
    if not path.exists():
        return [], []

    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    parsed_docs = [doc for doc in docs if doc is not None]
    if not parsed_docs:
        return [], []

    ids_seen: set[int] = set()
    max_id = -1
    normalized: list[dict[str, Any]] = []

    for doc in parsed_docs:
        if not isinstance(doc, dict):
            raise ValueError("database YAML entries must be mappings")
        if "id" not in doc or "body" not in doc:
            raise ValueError("database YAML entries require 'id' and 'body'")

        doc_id = doc["id"]
        body = doc["body"]
        metadata = doc.get("metadata")

        if not isinstance(doc_id, int) or doc_id < 0:
            raise ValueError("database YAML entry 'id' must be a non-negative integer")
        if doc_id in ids_seen:
            raise ValueError(f"database YAML has duplicate id {doc_id}")
        if not isinstance(body, str):
            raise ValueError(f"database YAML entry body for id {doc_id} must be a string")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"database YAML entry metadata for id {doc_id} must be a mapping")

        ids_seen.add(doc_id)
        max_id = max(max_id, doc_id)
        normalized.append({"id": doc_id, "body": body, "metadata": metadata})

    texts = [""] * (max_id + 1)
    metas: list[dict[str, Any] | None] = [None] * (max_id + 1)
    for rec in normalized:
        rid = rec["id"]
        texts[rid] = rec["body"]
        metas[rid] = rec["metadata"]

    return texts, metas


def save_yaml_tables(path: Path, texts: list[str], metas: list[dict[str, Any] | None]) -> None:
    docs: list[dict[str, Any]] = []
    for doc_id, body in enumerate(texts):
        rec: dict[str, Any] = {
            "id": doc_id,
            "metadata": metas[doc_id] if doc_id < len(metas) and metas[doc_id] is not None else {},
            "body": LiteralString(body),
        }
        docs.append(rec)

    payload = yaml.safe_dump_all(
        docs,
        explicit_start=True,
        sort_keys=False,
        allow_unicode=True,
    )
    path.write_text(payload, encoding="utf-8")


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return np.zeros_like(v)
    return v / n


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_blank_body(text: str | None) -> bool:
    return text is None or normalize_whitespace(text) == ""


def is_deleted_record(metadata: dict[str, Any] | None, body: str | None) -> bool:
    if isinstance(metadata, dict) and bool(metadata.get("deleted")):
        return True
    if body is None:
        return False
    try:
        parsed = yaml.safe_load(body)
    except Exception:
        return False
    return isinstance(parsed, dict) and bool(parsed.get("deleted"))


def embed_text_hash(text: str, dim: int = DIM) -> np.ndarray:
    normalized = normalize_whitespace(text)
    tokens = re.findall(r"[a-zA-Z0-9_]+", normalized.lower())
    vec = np.zeros((dim,), dtype=np.float32)
    for token in tokens:
        h = hash(token)
        idx = abs(h) % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += sign
    return normalize(vec).astype(np.float32)


def parse_yaml_flow_map(expr: str) -> dict[str, Any]:
    parsed = yaml.safe_load(expr)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("filter expression must parse to a YAML mapping")
    return parsed


def compare_values(lhs: Any, rhs: Any) -> int:
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return 0
    lhs_s = str(lhs)
    rhs_s = str(rhs)
    if lhs_s < rhs_s:
        return -1
    if lhs_s > rhs_s:
        return 1
    return 0


def bare_equals(value: Any, expected: Any) -> bool:
    if isinstance(value, list):
        return any(str(v) == str(expected) for v in value)
    return str(value) == str(expected)


def eval_condition(data: dict[str, Any], key: str, cond: Any) -> bool:
    if key not in data:
        return False
    value = data[key]

    if isinstance(cond, dict):
        if len(cond) != 1:
            return False
        op, operand = next(iter(cond.items()))
        if op == "$gte":
            return compare_values(value, operand) >= 0
        if op == "$lte":
            return compare_values(value, operand) <= 0
        if op == "$ne":
            return not bare_equals(value, operand)
        if op == "$prefix":
            return isinstance(value, str) and str(value).startswith(str(operand))
        if op == "$contains":
            return isinstance(value, list) and any(str(v) == str(operand) for v in value)
        return False

    return bare_equals(value, cond)


def matches_filter(data: dict[str, Any], filt: dict[str, Any]) -> bool:
    for key, cond in filt.items():
        if key == "$and":
            if not isinstance(cond, list):
                return False
            if not all(isinstance(c, dict) and matches_filter(data, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not isinstance(cond, list):
                return False
            if not any(isinstance(c, dict) and matches_filter(data, c) for c in cond):
                return False
            continue
        if not eval_condition(data, key, cond):
            return False
    return True


def create_index() -> faiss.IndexIDMap2:
    base = faiss.IndexHNSWFlat(DIM, 32)
    base.hnsw.efConstruction = 200
    base.hnsw.efSearch = 64
    return faiss.IndexIDMap2(base)


def load_index(path: Path, verbose: bool) -> faiss.IndexIDMap2:
    if not path.exists():
        return create_index()
    try:
        idx = faiss.read_index(str(path))
    except Exception:
        return create_index()
    if isinstance(idx, faiss.IndexIDMap2):
        return idx
    wrapped = faiss.IndexIDMap2(idx)
    vlog(verbose, "Loaded non-IDMap index; wrapping in IndexIDMap2")
    return wrapped


def get_existing_ids(index: faiss.IndexIDMap2) -> set[int]:
    if index.ntotal == 0:
        return set()
    arr = faiss.vector_to_array(index.id_map)
    return set(int(x) for x in arr.tolist())


def rebuild_index_from_texts(texts: list[str | None], verbose: bool) -> faiss.IndexIDMap2:
    idx = create_index()
    indexed = 0
    skipped_blank = 0
    for doc_id, text in enumerate(texts):
        note = text or ""
        if is_blank_body(note):
            skipped_blank += 1
            continue
        vec = embed_text_hash(note)
        idx.add_with_ids(vec.reshape(1, -1), np.array([doc_id], dtype=np.int64))
        indexed += 1
    vlog(verbose, f"Rebuilt index with {indexed} vectors (skipped {skipped_blank} blank records)")
    return idx


def search_all(index: faiss.IndexIDMap2, query_vec: np.ndarray) -> list[Result]:
    if index.ntotal == 0:
        return []
    k = int(index.ntotal)
    scores, ids = index.search(query_vec.reshape(1, -1), k)
    out: list[Result] = []
    for s, doc_id in zip(scores[0].tolist(), ids[0].tolist()):
        if doc_id < 0:
            continue
        out.append(Result(int(doc_id), float(s)))
    return out


def print_recall_result_multiline(rank: int, score: float, text: str) -> None:
    print(f"  [{rank}] Score: {score:.4f} |")
    lines = text.splitlines() or [""]
    for ln in lines:
        print(f"      {ln}")


def command_clean(db_base: str, user_cwd: str) -> int:
    index_path, yaml_path = build_db_paths(db_base, user_cwd)
    removed_any = False
    for p in (index_path, yaml_path):
        try:
            p.unlink()
            removed_any = True
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"Error: failed to remove {p}: {e}", file=sys.stderr)
            return 1

    if removed_any:
        print(
            "Cleared memory database "
            f"({index_path}, {yaml_path})"
        )
    else:
        print(
            "Database already empty "
            f"({index_path}, {yaml_path})"
        )
    return 0


def command_reindex(db_base: str, user_cwd: str, verbose: bool) -> int:
    index_path, yaml_path = build_db_paths(db_base, user_cwd)

    try:
        texts, metas = load_yaml_tables(yaml_path)
    except Exception as e:
        print(f"Error: failed to load database YAML '{yaml_path}': {e}", file=sys.stderr)
        return 1

    # Compact records before rebuild: drop blank/deleted entries and re-sequence IDs.
    compact_texts: list[str] = []
    compact_metas: list[dict[str, Any] | None] = []
    dropped = 0
    for i, text in enumerate(texts):
        metadata = metas[i] if i < len(metas) else None
        if is_blank_body(text) or is_deleted_record(metadata, text):
            dropped += 1
            continue
        compact_texts.append(text)
        compact_metas.append(metadata)

    # Canonicalize YAML formatting and persist compacted IDs on reindex.
    ensure_parent_dir(yaml_path)
    save_yaml_tables(yaml_path, compact_texts, compact_metas)

    index = rebuild_index_from_texts(compact_texts, verbose)
    ensure_parent_dir(index_path)
    faiss.write_index(index, str(index_path))
    print(f"Rebuilt index from {yaml_path.name}")
    print(f"Wrote index: {index_path.name}")
    if dropped > 0:
        print(f"Compacted: dropped {dropped} blank/deleted entries")
    return 0


def parse_save_yaml_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"failed to read input file '{path}'")
    text = path.read_text(encoding="utf-8")
    docs = list(yaml.safe_load_all(text))
    entries: list[dict[str, Any]] = []

    for doc in docs:
        if doc is None:
            continue
        if not isinstance(doc, dict):
            raise ValueError("each YAML document must be a mapping")
        if "body" not in doc:
            raise ValueError("each YAML document requires 'body'")
        body = doc.get("body")
        if not isinstance(body, str) or body.strip() == "":
            raise ValueError("body must be a non-empty string")

        metadata = doc.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a mapping when provided")

        rec: dict[str, Any] = {"body": body, "metadata": metadata}
        if "id" in doc:
            if not isinstance(doc["id"], int) or doc["id"] < 0:
                raise ValueError("id must be a non-negative integer when provided")
            rec["id"] = int(doc["id"])
        entries.append(rec)

    if not entries:
        raise ValueError("input YAML contains no entries")
    return entries


def command_save(db_base: str, save_yaml_path: str, user_cwd: str, verbose: bool) -> int:
    index_path, yaml_path = build_db_paths(db_base, user_cwd)
    entries = parse_save_yaml_file(Path(save_yaml_path))

    try:
        texts, metas = load_yaml_tables(yaml_path)
    except Exception as e:
        print(f"Error: failed to load database YAML '{yaml_path}': {e}", file=sys.stderr)
        return 1

    if len(metas) < len(texts):
        metas.extend([None] * (len(texts) - len(metas)))

    index = load_index(index_path, verbose)
    existing_ids = get_existing_ids(index)
    had_overwrite = False

    for entry in entries:
        note = entry["body"]
        metadata = entry.get("metadata")

        override_id = entry.get("id")
        if override_id is not None:
            if override_id >= len(texts) or override_id not in existing_ids:
                print(f"Error: override id {override_id} does not exist", file=sys.stderr)
                return 1

            texts[override_id] = note
            metas[override_id] = metadata
            had_overwrite = True
            print(f"Memorized: '{note}' (ID: {override_id})")
        else:
            new_id = len(texts)
            vec = embed_text_hash(note)
            index.add_with_ids(vec.reshape(1, -1), np.array([new_id], dtype=np.int64))
            texts.append(note)
            metas.append(metadata)
            print(f"Memorized: '{note}' (ID: {new_id})")

    if had_overwrite:
        index = rebuild_index_from_texts(texts, verbose)

    ensure_parent_dir(index_path)
    ensure_parent_dir(yaml_path)

    faiss.write_index(index, str(index_path))
    save_yaml_tables(yaml_path, texts, metas)
    return 0


def command_recall(db_base: str, query: str, k: int, filter_expr: str | None, user_cwd: str) -> int:
    index_path, yaml_path = build_db_paths(db_base, user_cwd)

    try:
        texts, metas = load_yaml_tables(yaml_path)
    except Exception as e:
        print(f"Error: failed to load database YAML '{yaml_path}': {e}", file=sys.stderr)
        return 1

    index = load_index(index_path, verbose=False)

    print(f"Top {k} results for '{query}':")
    if index.ntotal == 0:
        return 0

    query_vec = embed_text_hash(query)
    all_results = search_all(index, query_vec)

    active_filter: dict[str, Any] | None = None
    if filter_expr is not None:
        try:
            active_filter = parse_yaml_flow_map(filter_expr)
        except Exception as e:
            print(f"Error: invalid --filter expression: {e}", file=sys.stderr)
            return 1

    shown = 0
    for result in all_results:
        if shown >= k:
            break
        if result.score < -0.9:
            continue

        doc_id = result.doc_id
        if doc_id < 0 or doc_id >= len(texts):
            continue

        if active_filter is not None:
            record = metas[doc_id] if doc_id < len(metas) and metas[doc_id] is not None else {}
            if not record:
                continue
            if not matches_filter(record, active_filter):
                continue

        text = texts[doc_id] or ""
        if is_blank_body(text):
            continue
        print_recall_result_multiline(shown + 1, result.score, text)
        shown += 1

    return 0


def parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def resolve_field_value(doc_id: int, metadata: dict[str, Any], field: str) -> Any:
    if field == "id":
        return doc_id
    if field == "metadata":
        return metadata
    key = field[len("metadata.") :] if field.startswith("metadata.") else field
    return metadata.get(key)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
    return str(value)


def default_analyze_fields(matches: list[tuple[int, dict[str, Any]]]) -> list[str]:
    keys: set[str] = set()
    for _, metadata in matches:
        keys.update(str(k) for k in metadata.keys())
    extra = sorted(keys)[:3]
    return ["id", *extra]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not headers:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def print_stats(matches: list[tuple[int, dict[str, Any]]], key: str) -> None:
    values: list[Any] = []
    for doc_id, metadata in matches:
        value = resolve_field_value(doc_id, metadata, key)
        if value is not None:
            values.append(value)

    counter: Counter[str] = Counter(format_cell(v) for v in values)
    print(f"Key: {key}")
    print(f"Cardinality (distinct values): {len(counter)}")
    print("Cardinality by value:")
    top = counter.most_common(4)
    for name, count in top:
        print(f"  {name}: {count}")
    if len(counter) > 4:
        other = sum(counter.values()) - sum(c for _, c in top)
        print(f"  other (aggregate of {len(counter) - 4} additional values): {other}")

    if values:
        numeric: list[float] = []
        numeric_ok = True
        for value in values:
            if isinstance(value, (int, float)):
                numeric.append(float(value))
                continue
            try:
                numeric.append(float(str(value)))
            except (ValueError, TypeError):
                numeric_ok = False
                break

        if numeric_ok and numeric:
            avg = sum(numeric) / len(numeric)
            print("Range (numeric):")
            print(f"  min: {min(numeric):g}")
            print(f"  max: {max(numeric):g}")
            print(f"  avg: {avg:.2f}")
            return

        dates: list[datetime] = []
        date_ok = True
        for value in values:
            parsed = parse_iso_datetime(value)
            if parsed is None:
                date_ok = False
                break
            dates.append(parsed)
        if date_ok and dates:
            start = min(dates)
            end = max(dates)
            print("Range (date-like):")
            print(f"  start: {start.date().isoformat()}")
            print(f"  end:   {end.date().isoformat()}")


def command_analyze(
    db_base: str,
    filter_expr: str,
    fields: list[str] | None,
    stats_key: str | None,
    limit: int,
    offset: int,
    user_cwd: str,
) -> int:
    if not filter_expr.strip():
        print("Error: analyze requires --filter <expr>", file=sys.stderr)
        return 1
    if limit < 1:
        print("Error: --limit must be >= 1", file=sys.stderr)
        return 1
    if offset < 0:
        print("Error: --offset must be >= 0", file=sys.stderr)
        return 1

    _, yaml_path = build_db_paths(db_base, user_cwd)
    try:
        texts, metas = load_yaml_tables(yaml_path)
    except Exception as e:
        print(f"Error: failed to load database YAML '{yaml_path}': {e}", file=sys.stderr)
        return 1

    try:
        active_filter = parse_yaml_flow_map(filter_expr)
    except Exception as e:
        print(f"Error: invalid --filter expression: {e}", file=sys.stderr)
        return 1

    matches: list[tuple[int, dict[str, Any]]] = []
    for doc_id in range(len(texts)):
        metadata = metas[doc_id] if doc_id < len(metas) and metas[doc_id] is not None else {}
        if not metadata:
            continue
        if matches_filter(metadata, active_filter):
            matches.append((doc_id, metadata))

    print(f"Matched: {len(matches)}")
    if stats_key is not None:
        print_stats(matches, stats_key)
        return 0

    selected_fields = fields if fields else default_analyze_fields(matches)
    if not selected_fields:
        selected_fields = ["id"]

    page = matches[offset : offset + limit]
    rows: list[list[str]] = []
    for doc_id, metadata in page:
        row = [format_cell(resolve_field_value(doc_id, metadata, field)) for field in selected_fields]
        rows.append(row)

    headers = ["ID" if field == "id" else field for field in selected_fields]
    print_table(headers, rows)
    return 0


def print_help() -> None:
    print("Usage:")
    print("  memo --help")
    print("  memo -f <base> [-v] save <yaml_file>")
    print("  memo -f <base> [-v] recall [-k <N>] [--filter <expr>] <query>")
    print("  memo -f <base> [-v] analyze --filter <expr> [--fields <list>] [--stats <key>] [--limit <N>] [--offset <N>]")
    print("  memo -f <base> [-v] clean")
    print("  memo -f <base> [-v] reindex")
    print()
    print("Commands:")
    print("  save                Insert/update memory records from YAML input file")
    print("  recall              Semantic recall from <base>.memo + <base>.yaml")
    print("  analyze             Metadata-only reporting from <base>.yaml")
    print("  clean               Remove <base>.memo and <base>.yaml")
    print("  reindex             Rebuild <base>.memo from <base>.yaml (full regenerate)")
    print()
    print("Options:")
    print("  -f <base>           REQUIRED DB basename")
    print("  -v                 Verbose logs to stderr")
    print("  <yaml_file>        YAML file for save input (single or multi-doc using ---)")
    print("                     Each doc requires: metadata: <map>, body: <string>")
    print("                     Optional per-doc id: <int> to overwrite existing record")
    print("  --filter <expr>    Filter recall results by metadata")
    print("  --fields <list>    analyze only: comma-separated columns (e.g. id,source,metadata)")
    print("  --stats <key>      analyze only: cardinality + numeric/date-like range for key")
    print("  --limit <N>        analyze only: max rows to print (default: 100)")
    print("  --offset <N>       analyze only: rows to skip before printing (default: 0)")
    print("  --help             Show this help")


def parse_args(argv: list[str]) -> tuple[dict[str, Any], int]:
    db_base: str | None = None
    verbose = False
    positional: list[str] = []

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-v":
            verbose = True
            i += 1
            continue
        if arg == "-f":
            if i + 1 >= len(argv):
                print("Error: -f requires a value", file=sys.stderr)
                return {}, 1
            db_base = argv[i + 1]
            if db_base.strip() == "":
                print("Error: -f requires a non-empty value", file=sys.stderr)
                return {}, 1
            i += 2
            continue
        positional.append(arg)
        i += 1

    return {
        "db_base": db_base,
        "verbose": verbose,
        "positional": positional,
    }, 0


def parse_recall_args(args: list[str]) -> tuple[dict[str, Any], int]:
    k = 2
    filter_expr: str | None = None
    query_parts: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-k":
            if i + 1 >= len(args):
                print("Error: -k requires an integer", file=sys.stderr)
                return {}, 1
            try:
                k = int(args[i + 1])
            except ValueError:
                print("Error: -k requires an integer", file=sys.stderr)
                return {}, 1
            i += 2
            continue
        if arg == "--filter":
            if i + 1 >= len(args):
                print("Error: --filter requires a filter expression", file=sys.stderr)
                return {}, 1
            filter_expr = args[i + 1]
            i += 2
            continue
        query_parts.append(arg)
        i += 1

    query = " ".join(query_parts).strip()
    if not query:
        print("Error: recall requires <query>", file=sys.stderr)
        return {}, 1

    if k < 1:
        k = 1
    if k > MAX_K:
        k = MAX_K

    return {"k": k, "filter_expr": filter_expr, "query": query}, 0


def parse_analyze_args(args: list[str]) -> tuple[dict[str, Any], int]:
    filter_expr: str | None = None
    fields: list[str] | None = None
    stats_key: str | None = None
    limit = 100
    offset = 0

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--filter":
            if i + 1 >= len(args):
                print("Error: --filter requires a filter expression", file=sys.stderr)
                return {}, 1
            filter_expr = args[i + 1]
            i += 2
            continue
        if arg == "--fields":
            if i + 1 >= len(args):
                print("Error: --fields requires a comma-separated field list", file=sys.stderr)
                return {}, 1
            parsed_fields = [f.strip() for f in args[i + 1].split(",") if f.strip()]
            if not parsed_fields:
                print("Error: --fields requires at least one field", file=sys.stderr)
                return {}, 1
            fields = parsed_fields
            i += 2
            continue
        if arg == "--stats":
            if i + 1 >= len(args):
                print("Error: --stats requires a key", file=sys.stderr)
                return {}, 1
            stats_key = args[i + 1].strip()
            if not stats_key:
                print("Error: --stats requires a non-empty key", file=sys.stderr)
                return {}, 1
            i += 2
            continue
        if arg == "--limit":
            if i + 1 >= len(args):
                print("Error: --limit requires an integer", file=sys.stderr)
                return {}, 1
            try:
                limit = int(args[i + 1])
            except ValueError:
                print("Error: --limit requires an integer", file=sys.stderr)
                return {}, 1
            i += 2
            continue
        if arg == "--offset":
            if i + 1 >= len(args):
                print("Error: --offset requires an integer", file=sys.stderr)
                return {}, 1
            try:
                offset = int(args[i + 1])
            except ValueError:
                print("Error: --offset requires an integer", file=sys.stderr)
                return {}, 1
            i += 2
            continue

        print(f"Error: unknown analyze option '{arg}'", file=sys.stderr)
        return {}, 1

    if filter_expr is None:
        print("Error: analyze requires --filter <expr>", file=sys.stderr)
        return {}, 1

    return {
        "filter_expr": filter_expr,
        "fields": fields,
        "stats_key": stats_key,
        "limit": limit,
        "offset": offset,
    }, 0


def main() -> int:
    parsed, rc = parse_args(sys.argv)
    if rc != 0:
        return rc

    positional = parsed["positional"]
    if not positional or positional[0] in {"--help", "help"}:
        print_help()
        return 0

    user_cwd = os.getcwd()
    command = positional[0]
    db_base = parsed["db_base"]
    if db_base is None:
        print("Error: -f <base> is required", file=sys.stderr)
        print_help()
        return 1
    verbose = parsed["verbose"]

    if command == "clean":
        if len(positional) != 1:
            print("Error: clean does not accept extra arguments", file=sys.stderr)
            return 1
        return command_clean(db_base, user_cwd)

    if command == "reindex":
        if len(positional) != 1:
            print("Error: reindex does not accept extra arguments", file=sys.stderr)
            return 1
        return command_reindex(db_base, user_cwd, verbose)

    if command == "save":
        if len(positional) != 2:
            print("Error: save requires exactly one <yaml_file>", file=sys.stderr)
            return 1
        return command_save(db_base, positional[1], user_cwd, verbose)

    if command == "recall":
        recall_args, recall_rc = parse_recall_args(positional[1:])
        if recall_rc != 0:
            return recall_rc
        return command_recall(
            db_base,
            recall_args["query"],
            recall_args["k"],
            recall_args["filter_expr"],
            user_cwd,
        )

    if command == "analyze":
        analyze_args, analyze_rc = parse_analyze_args(positional[1:])
        if analyze_rc != 0:
            return analyze_rc
        return command_analyze(
            db_base,
            analyze_args["filter_expr"],
            analyze_args["fields"],
            analyze_args["stats_key"],
            analyze_args["limit"],
            analyze_args["offset"],
            user_cwd,
        )

    print(f"Error: unknown command '{command}'", file=sys.stderr)
    print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
