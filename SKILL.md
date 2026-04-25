# memo — Usage Guide for AI Agents

`memo` is a FAISS-backed local vector memory tool for storing and recalling long-term facts.

## Command help

```text
Usage:
  memo --help
  memo -f <base> [-v] save <yaml_file>
  memo -f <base> [-v] recall [-k <N>] [--filter <expr>] <query>
  memo -f <base> [-v] analyze --filter <expr> [--fields <list>] [--stats <key>] [--limit <N>] [--offset <N>]
  memo -f <base> [-v] clean
  memo -f <base> [-v] reindex

Commands:
  save                Insert/update memory records from YAML input file
  recall              Semantic recall from <base>.memo + <base>.yaml
  analyze             Metadata-only reporting from <base>.yaml
  clean               Remove <base>.memo and <base>.yaml
  reindex             Rebuild <base>.memo from <base>.yaml (full regenerate)

Options:
  -f <base>           REQUIRED DB basename
  -v                 Verbose logs to stderr
  <yaml_file>        YAML file for save input (single or multi-doc using ---)
                     Each doc requires: metadata: <map>, body: <string>
                     Optional per-doc id: <int> to overwrite existing record
  --filter <expr>    Filter recall results by metadata
  --fields <list>    analyze only: comma-separated columns (e.g. id,source,metadata)
  --stats <key>      analyze only: cardinality + numeric/date-like range for key
  --limit <N>        analyze only: max rows to print (default: 100)
  --offset <N>       analyze only: rows to skip before printing (default: 0)
  --help             Show this help
```

## Core behavior

- `memo --help` prints help.
- `-f <base>` is required for all subcommands.
- `memo -f <base> save <yaml_file>` is the only supported save input mode.
- YAML input can contain one or many docs (`---` separators).
- Each YAML doc requires:
  - `body` (non-empty string)
  - optional `metadata` (mapping)
  - optional `id` (non-negative integer) to overwrite an existing record.
- `memo -f <base> recall <query>` recalls top matches (default `k=2`).
- `memo -f <base> recall -k <N> <query>` recalls top `N` matches (`N` capped at 100).
- `memo -f <base> recall --filter '<expr>' <query>` filters on metadata using YAML-flow expressions/operators.
- `memo -f <base> analyze --filter '<expr>'` runs metadata-only analysis (no semantic query).
- `memo -f <base> analyze --stats <key>` prints cardinality and numeric/date-like range summaries.
- `memo -f <base> analyze --fields id,source,...` projects metadata rows without body text.
- `memo -f <base> clean` wipes the current DB files (`<base>.memo`, `<base>.yaml`).
- `memo -f <base> reindex` rebuilds `<base>.memo` from `<base>.yaml` without editing YAML.
- Relative `-f` paths resolve from process CWD.
- `-v` enables verbose logs to stderr only.

## Save input format

Single document:

```yaml
metadata:
  source: user
  tags: [prefs, profile]
body: |
  My favorite color is blue.
```

Batch documents:

```yaml
---
metadata:
  source: user
  category: health
body: I am allergic to peanuts.

---
metadata:
  source: chat
  category: pref
body: User prefers dark mode.
```

Overwrite existing id:

```yaml
---
id: 3
metadata:
  source: user
body: Updated note text for id 3.
```

## Real examples

### Save + recall

```bash
$ memo -f memo save /tmp/memo-input.yaml
Memorized: 'I am allergic to peanuts.' (ID: 0)
Memorized: 'User prefers dark mode.' (ID: 1)

$ memo -f memo recall -k 2 health info
Top 2 results for 'health info':
  [1] Score: 0.2300 |
      I am allergic to peanuts.
```

### Filtered recall

```bash
$ memo -f memo recall -k 3 --filter '{source: user}' what do I know about myself
Top 3 results for 'what do I know about myself':
  [1] Score: 0.2500 |
      I am allergic to peanuts.
```

### Analyze

```bash
$ memo -f memo analyze --filter '{source: user}' --fields id,source,category
Matched: 1
ID  source  category
0   user    health
```

### Clean

```bash
$ memo -f memo clean
Cleared memory database (memo.memo, memo.yaml)
```

### Reindex

```bash
$ memo -f memo reindex
Rebuilt index from memo.yaml
Wrote index: memo.memo
```

## Output contract

- Normal mode: only subcommand result text goes to stdout.
- Recall output uses a multi-line block format:
  - header line with rank and score
  - note body lines indented below
- Verbose mode (`-v`): debug/startup logs on stderr.

## Important differences vs legacy C memo

- No `memo save <note>` positional note mode.
- No stdin save mode (`memo save -`).
- No `-m` / `-i` interleaved batch mode.
- YAML file input is required for all saves.
- Runtime storage uses `.memo + .yaml`.

## Metadata filtering (embedded reference)

`memo -f <base> recall --filter '<expr>' <query>` applies deterministic metadata filtering
before showing semantic results.

### Metadata shape (save YAML)

Metadata is stored per record from the `metadata` mapping in each YAML document:

```yaml
---
metadata:
  source: user
  ts: 2026-02-21
  tags: [personal, food]
  priority: 2
body: I love sushi.
```

Records without `metadata` do not match any `--filter` expression.

### Filter syntax

Filters use YAML Flow syntax. Outer `{}` are optional.

- `source: user`
- `{source: user}`

Both are equivalent.

### Supported operators

| Operator       | Meaning            | Example                      |
|----------------|--------------------|------------------------------|
| *(bare value)* | exact match        | `source: user`               |
| `$ne`          | not equal          | `source: {$ne: system}`      |
| `$gte`         | greater-or-equal   | `priority: {$gte: 2}`        |
| `$lte`         | less-or-equal      | `ts: {$lte: 2026-01-31}`     |
| `$prefix`      | prefix match       | `category: {$prefix: per}`   |
| `$contains`    | array contains     | `tags: {$contains: food}`    |

### Boolean logic

Top-level keys are implicitly ANDed:

```bash
memo -f memo recall --filter 'source: user, priority: {$gte: 2}' urgent user items
```

Use `$and` for multiple conditions on the same key:

```bash
memo -f memo recall --filter '$and: [{ts: {$gte: 2026-01-01}}, {ts: {$lte: 2026-01-31}}]' january memories
```

Use `$or` for alternative conditions:

```bash
memo -f memo recall --filter '$or: [{source: user}, {source: chat}]' user or chat memories
```

### Practical examples

```bash
# Exact match
memo -f memo recall --filter 'source: chat' what did we talk about

# Numeric threshold
memo -f memo recall --filter 'priority: {$gte: 3}' important things

# Lexicographic/date threshold
memo -f memo recall --filter 'ts: {$gte: 2026-02-01}' recent events

# Prefix
memo -f memo recall --filter 'category: {$prefix: per}' personal items

# Array contains
memo -f memo recall --filter 'tags: {$contains: food}' what food do I like

# Combined conditions
memo -f memo recall --filter 'source: user, category: health, priority: {$gte: 2}' important health
```

### How filtering is applied

1. Metadata filter evaluates across records and determines candidate IDs.
2. Vector search runs on those candidates.
3. Results are returned in score order.

If `--filter` is omitted, all records are eligible for vector search.
