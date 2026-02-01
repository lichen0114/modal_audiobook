# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EPUB to MP3 audiobook converter using Modal (cloud GPU) with three TTS models:
- **Kokoro-82M** (default): Lightweight 82M model, 54 voices across 9 languages, runs on T4 GPU
- **Qwen3-TTS**: Multi-speaker, multi-language, supports style instructions, runs on A10G GPU
- **Chatterbox**: English-only, single default voice, runs on A10G GPU

## Commands

```bash
# Install dependencies
pip install -e .

# Deploy Modal TTS service (required before conversions)
python3 -m modal deploy modal_app/tts_service.py

# Test TTS service
python3 -m modal run modal_app/tts_service.py                      # Kokoro (default)
python3 -m modal run modal_app/tts_service.py --model qwen         # Qwen
python3 -m modal run modal_app/tts_service.py --model chatterbox   # Chatterbox

# Convert EPUB to audiobook
python main.py book.epub                                           # Kokoro default
python main.py book.epub -s bf_alice -l "British English"          # Kokoro British
python main.py book.epub --model qwen -s Ryan --instruct "Read calmly"
python main.py book.epub --no-resume                               # Ignore previous progress

# List available voices
python main.py --list-speakers
python main.py --list-speakers --model qwen
```

## Architecture

```
Local (main.py)                        Modal Cloud (GPU)
├── EPUB parsing (ebooklib)            ├── KokoroTTSService (T4 GPU, max 8 containers)
├── Text chunking (sentence-aware)     ├── TTSService (Qwen, A10G GPU, max 5 containers)
├── Parallel chapter processing        └── ChatterboxTTSService (A10G GPU, max 5 containers)
└── Thread-safe progress tracking
```

**Data flow**: EPUB → chapters → text chunks (≤2000 chars) → Modal TTS → MP3 bytes → local files

**Parallelization**:
- Chapters processed concurrently (4 workers via ThreadPoolExecutor)
- Batches within chapters processed in parallel via Modal's `.starmap()`
- Thread-safe progress tracking with `ThreadSafeProgress` class

**Key design decisions**:
- Text chunked at sentence boundaries for natural speech
- Paragraph endings get longer pauses (800ms vs 500ms)
- Progress tracked in `.progress.json` for resume (includes model to prevent mixing)
- Model weights cached in Modal Volume (`audiobook-model-cache`)
- Containers auto-scale down after 2 min idle (`scaledown_window=120`)

## Module Responsibilities

- `main.py`: CLI, orchestration, parallel processing, thread-safe progress tracking
- `src/epub_parser.py`: Chapter extraction from EPUB spine, filters TOC/cover
- `src/text_chunker.py`: Sentence-boundary splitting, paragraph end tracking
- `src/config.py`: Model config, speakers/languages, tunable constants
- `modal_app/tts_service.py`: Modal TTS service classes with GPU/container config

## Adding a New TTS Model

1. **`src/config.py`**: Add voice/language constants, update `MODELS` list and `MODEL_CONFIG`
2. **`modal_app/tts_service.py`**: Add `create_<model>_image()`, create service class with `synthesize_chunk`/`synthesize_chapter` methods, configure `@app.cls` with appropriate GPU, `scaledown_window`, and `max_containers`
3. **`main.py`**: Update `get_tts_service()`, add validation logic, handle language code conversion if needed

## Configuration

| Model | Voices | Languages | Style Instructions | GPU |
|-------|--------|-----------|-------------------|-----|
| kokoro (default) | 54 | 9 | No | T4 |
| qwen | 9 | 10 | Yes (`--instruct`) | A10G |
| chatterbox | 1 (default) | English | No | A10G |

**Kokoro voice prefixes**: af/am=American, bf/bm=British, jf/jm=Japanese, zf/zm=Chinese, ef/em=Spanish, ff=French, hf/hm=Hindi, if/im=Italian, pf/pm=Portuguese

**Tunable constants** in `src/config.py`:
- `MAX_CHUNK_LENGTH` (2000): Max chars per TTS chunk
- `MAX_CHUNKS_PER_BATCH` (30): Max chunks per Modal call
- `MAX_PARALLEL_CHAPTERS` (4): Concurrent chapter workers
- `PAUSE_BETWEEN_CHUNKS_MS` (500), `PAUSE_BETWEEN_PARAGRAPHS_MS` (800)
