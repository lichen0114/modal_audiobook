# 🎧 Modal Audiobook

> **Convert EPUB books to high-quality MP3 audiobooks using cloud GPUs**

A Python-based tool that transforms EPUB files into audiobooks using state-of-the-art TTS models powered by [Modal](https://modal.com/) cloud GPUs.

## ✨ Features

- 🚀 **Parallel Processing** - Process 4 chapters simultaneously
- 🔄 **Resume Support** - Continue interrupted conversions
- 📚 **Smart Chunking** - Sentence-boundary aware text splitting
- 🎙️ **54+ Voices** - Multiple voices across 9 languages
- ☁️ **Auto-scaling** - Cloud GPU scales on demand

## 🎯 Supported Models

| Model | Size | GPU | Voices | Languages |
|-------|------|-----|--------|-----------|
| **Kokoro-82M** (default) | 82M | T4 | 54 | 9 |
| **Qwen3-TTS** | 1.7B | A10G | 9 | 10 |
| **Chatterbox** | - | A10G | 1 | English |

## 🚀 Quick Start

### Prerequisites

1. Install [Modal](https://modal.com/) and authenticate:
   ```bash
   pip install modal
   modal setup
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Deploy the TTS service:
   ```bash
   modal deploy modal_app/tts_service.py
   ```

### Convert an EPUB

```bash
# Basic conversion (uses Kokoro with default voice)
python main.py "My Book.epub"

# Custom voice
python main.py "My Book.epub" -s bf_alice -l "British English"

# Using Qwen with style instructions
python main.py "My Book.epub" --model qwen --instruct "Read slowly and dramatically"
```

### List Available Voices

```bash
python main.py --list-speakers              # Kokoro voices
python main.py --list-speakers --model qwen # Qwen voices
```

## 🎤 Voice Examples

### Kokoro Voices

| Language | Female | Male |
|----------|--------|------|
| American English | `af_heart`, `af_bella` | `am_adam`, `am_michael` |
| British English | `bf_alice`, `bf_emma` | `bm_george`, `bm_lewis` |
| Japanese | `jf_alpha`, `jf_gongitsune` | `jm_kumo` |
| Chinese | `zf_xiaobei`, `zf_xiaoni` | `zm_yunxi`, `zm_yunjian` |

### Qwen Voices

`Chelsie`, `Ethan`, `Aiden`, `Ryan` (default), `Emily`, `Vivian`, `Bella`, `Serena`, `Aurora`

## 📁 Output Structure

```
audiobooks/
└── My Book/
    ├── 001_Chapter_1.mp3
    ├── 002_Chapter_2.mp3
    ├── ...
    └── .progress.json  # Resume tracking
```

## ⚙️ CLI Options

```
usage: main.py [-h] [-o OUTPUT] [-m {kokoro,qwen,chatterbox}] 
               [-s SPEAKER] [-l LANGUAGE] [--instruct INSTRUCT] 
               [--no-resume] [--list-speakers] [epub]

Options:
  -o, --output       Output directory (default: ./audiobooks)
  -m, --model        TTS model: kokoro, qwen, chatterbox
  -s, --speaker      Voice to use (model-specific)
  -l, --language     Language for TTS
  --instruct         Style instruction (Qwen only)
  --no-resume        Ignore previous progress
  --list-speakers    List available voices
```

## 📖 Documentation

See [CODEBASE.md](CODEBASE.md) for detailed technical documentation.

## 📝 License

MIT
