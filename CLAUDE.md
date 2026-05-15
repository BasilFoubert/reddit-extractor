# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_utils.py::test_filter_pp_by_urgency

# Lint
uv run ruff check src/

# Run a pipeline stage directly
uv run python src/ingest_posts.py
uv run python src/ingest_comments.py
uv run python src/build_threads.py
uv run python src/extract_pain_points.py
uv run python src/clustering.py
```

Always use `uv run python` — never bare `python`.

## Architecture

The pipeline is a linear sequence of scripts that each read/write JSONL files in `data/`:

```
data/raw/              ← downloaded from arctic-shift.photon-reddit.com
  r_ciso_posts.jsonl
  r_ciso_comments.jsonl

data/processed/
  r_ciso_posts.jsonl         ← ingest_posts.py      (field selection + normalize)
  r_ciso_comments.jsonl      ← ingest_comments.py   (filter short + normalize)
  r_ciso_threads.jsonl       ← build_threads.py     (join posts+comments into trees, score >= 3)
  r_ciso_pain_points.jsonl   ← extract_pain_points.py (LLM extraction, each post enriched with pain_points[])
  r_ciso_pain_points_clustered.jsonl ← clustering.py

data/embeddings/             ← clustering.py intermediate output
```

### Pain point extraction (`src/extract_pain_points.py`)

The most complex stage. Uses a **nested LangGraph workflow**:

- **Outer graph**: fans out over all threads in parallel (`spawn_post_workers` → `process_post`)
- **Inner graph** (one per post):
  1. `thread_scanner` — one LLM call to enumerate all pain points in the thread, returns `PostPainSummary` (list of `PainSummary` with index + 10-word description)
  2. `spawn_pain_workers` — fans out one worker per identified pain point
  3. `pain_point_extractor` — one LLM call per pain point to extract `verbatim`, `pain_point_reformulated`, and `urgency` (1–10)

Pain points are accumulated via `Annotated[list[PainPoint], add]` reducer. Rate limit errors trigger infinite retry with exponential backoff; other exceptions return empty results.

Model: `claude-haiku-4-5` (configurable via `Workflow.MODEL`).

### Shared types (`src/types.py`)

- `PainPoint` — Pydantic model, the core output unit
- `PainSummary` / `PostPainSummary` — intermediate LLM-structured output for the scan step
- `Comment` — TypedDict for the nested comment tree used by the extraction workflow

### Key data shapes

- Threads (`r_ciso_threads.jsonl`): post fields + `comments` as a recursive tree (`body`, `replies[]`)
- Pain points (`r_ciso_pain_points.jsonl`): thread fields + `pain_points[]` (verbatim, pain_point_reformulated, urgency, post_id)

### Clustering (`src/clustering.py`)

Embeds `pain_point_reformulated` text using `nomic-ai/nomic-embed-text-v1.5` (local, no API key), reduces dimensions with UMAP (40 components), then clusters with HDBSCAN. Uses MPS if available, falls back to CPU.

## Environment

Requires `ANTHROPIC_API_KEY` in `.env` (loaded via `python-dotenv`). The `.env` file is already present but not committed.

## Test data

`tests/data/` contains small fixture files for unit tests — use these when developing to avoid running full LLM pipelines. The `small_subreddit_pain_points.jsonl` fixture is pre-computed output from `extract_pain_points.py`.
