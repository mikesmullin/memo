# memo

| version | description |
|---|---|
| [v2](https://github.com/mikesmullin/memo/tree/v2) | Current FAISS-based reimplementation w/ YAML-based db. |
| [v1](https://github.com/mikesmullin/memo/tree/v1) | Previous C99/Vulkan implementation. |

FAISS-based reimplementation of `memo`.

## Goals

- Match existing `memo` CLI behaviors and output style while using modern local storage.
- Use FAISS for vector indexing/search.
- Use YAML as the human-readable source of records.
- Support full index regeneration from YAML.

## What It Is For

`memo` is a local semantic memory utility for:

- storing notes/facts in a YAML-backed database,
- recalling related notes by semantic similarity,
- filtering/analyzing records using metadata,
- rebuilding the vector index from YAML when needed.

## Installation

```bash
cd mac
uv sync
```

## Configuration

- Database basename is explicitly selected per invocation.
- For basename `<base>`, runtime files are:
  - `<base>.yaml` (human-readable records)
  - `<base>.memo` (FAISS index)
- Relative basenames are resolved from the process working directory.

## Record Format (YAML)

Single entry:

```yaml
metadata:
  source: chat
  tags: [ops, triage]
body: |
  A multi-line note body.
```

Batch entries in one file:

```yaml
---
metadata:
  source: chat
body: First note

---
metadata:
  source: user
body: Second note
```

Optional overwrite by id (feature parity with old `<id> <note>` path):

```yaml
---
id: 3
metadata:
  source: user
body: Updated note for id 3
```

For complete command usage, flags, and command input/output examples, see `SKILL.md`.
