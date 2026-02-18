# pdf2audiobook

Convert any PDF into a chapter-aware audiobook with a single command — or use the web app to upload, listen, and read along in real time.

## How it works

The pipeline runs four stages:

1. **Parse** — Extract chapters and structure from the PDF (auto-selects [Docling](https://github.com/DS4SD/docling) for complex layouts or PyMuPDF for simple ones)
2. **Clean** — Remove citations, tables, figures, and normalize text for speech using any LLM via [LiteLLM](https://github.com/BerriAI/litellm) (falls back to regex if unavailable)
3. **Chunk** — Split cleaned text into sentence-boundary-respecting chunks using spaCy
4. **Synthesize** — Generate audio with parallel TTS, stitch chapters, and produce a final M4B (with chapter markers) or MP3

Stages 2–4 run as a **streaming pipeline**: while one thread synthesizes Chapter 0 (CPU-heavy), another cleans Chapter 1 via LLM (I/O-heavy). Audio becomes available chapter-by-chapter instead of waiting for the entire book.

## Quick start

```bash
# Clone and install
git clone https://github.com/matt-ebrahim/pdf2audiobook.git
cd pdf2audiobook
pip install -e ".[kokoro,web]"

# Download the spaCy model
python -m spacy download en_core_web_sm

# Run via CLI
pdf2audiobook paper.pdf --tts kokoro --llm gpt-4o-mini

# Or launch the web app
pdf2audiobook-web
# Open http://localhost:8000
```

## CLI usage

```
pdf2audiobook <pdf> [options]

Options:
  -o, --output-dir DIR    Output directory (default: ./output/<pdf_name>)
  -c, --config FILE       Path to TOML config file
  --tts ENGINE            kokoro | chatterbox | openai | elevenlabs
  --llm MODEL             LiteLLM model string, or "none" to skip LLM cleaning
  --parser PARSER         auto | docling | pymupdf
  --voice NAME            Voice name (engine-specific)
  --format FORMAT         m4b | mp3
```

### Examples

```bash
# Local TTS with Kokoro, LLM cleaning with GPT-4o-mini
pdf2audiobook paper.pdf --tts kokoro --llm gpt-4o-mini

# API TTS with OpenAI, no LLM cleaning
pdf2audiobook book.pdf --tts openai --llm none --voice alloy

# Force Docling parser, output as MP3
pdf2audiobook scanned.pdf --parser docling --format mp3

# Resume an interrupted run (just re-run the same command)
pdf2audiobook paper.pdf --tts kokoro --llm gpt-4o-mini -o output/paper
```

Checkpoint/resume is automatic — if you interrupt a run, re-running the same command picks up where it left off.

## Web app

The web interface provides real-time streaming with:

- **Upload** — Drag-and-drop or click to upload any PDF
- **Live progress** — Watch chapters flow through cleaning → chunking → synthesis with color-coded status indicators
- **Instant playback** — Start listening as soon as the first chapter is ready, with auto-advance to the next
- **Read along** — Text reader panel shows the cleaned chapter text with paragraph highlighting synced to audio playback
- **PDF view** — Embedded PDF viewer tab to reference the original document
- **Downloads** — Download individual chapter MP3s or all chapters as a zip archive
- **Custom player** — Previous/next, seekable progress bar, playback speed control (0.5x–2x), keyboard shortcuts (Space, arrows)

```bash
pip install -e ".[kokoro,web]"
pdf2audiobook-web
```

## TTS engines

| Engine | Type | Install | Notes |
|--------|------|---------|-------|
| [Kokoro](https://github.com/hexgrad/kokoro) | Local | `pip install -e ".[kokoro]"` | Free, runs on CPU, good quality |
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | Local | `pip install -e ".[chatterbox]"` | Voice cloning support |
| OpenAI TTS | API | `pip install -e ".[openai-tts]"` | Set `OPENAI_API_KEY` |
| ElevenLabs | API | `pip install -e ".[elevenlabs]"` | Set `ELEVENLABS_API_KEY` |

## LLM cleaning

Text cleaning uses LiteLLM, which supports any LLM provider with a single model string:

```bash
--llm gpt-4o-mini                    # OpenAI
--llm claude-sonnet-4-20250514             # Anthropic
--llm gemini/gemini-2.0-flash        # Google
--llm ollama/llama3.2                # Local via Ollama
--llm none                           # Skip LLM, use regex only
```

Set the corresponding API key as an environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) or in the config file.

## Configuration

Copy `config.example.toml` and customize:

```bash
cp config.example.toml config.toml
pdf2audiobook paper.pdf -c config.toml
```

See [`config.example.toml`](config.example.toml) for all available options with descriptions.

## Project structure

```
src/pdf2audiobook/
├── cli.py              # Command-line interface
├── webapp.py           # FastAPI web application
├── pipeline.py         # Main pipeline orchestrator
├── streaming.py        # Streaming pipeline (concurrent chapter processing)
├── checkpoint.py       # Checkpoint/resume system
├── progress.py         # Progress reporting with SSE event support
├── config.py           # TOML configuration loading
├── models.py           # Data models (Chapter, ChunkMeta, etc.)
├── static/
│   └── index.html      # Web UI (single-page app)
├── parse/
│   ├── detector.py     # Auto-detect PDF complexity
│   ├── docling_parser.py   # Docling parser (complex PDFs)
│   └── pymupdf_parser.py   # PyMuPDF parser (simple PDFs)
├── clean/
│   └── cleaner.py      # LLM + regex text cleaning
├── chunk/
│   └── chunker.py      # spaCy sentence-boundary chunking
└── synth/
    ├── base.py         # Abstract TTS engine interface
    ├── synthesizer.py  # Synthesis orchestrator
    ├── stitcher.py     # Audio stitching & M4B creation
    ├── kokoro_tts.py   # Kokoro TTS backend
    ├── openai_tts.py   # OpenAI TTS backend
    ├── elevenlabs_tts.py   # ElevenLabs TTS backend
    └── chatterbox_tts.py   # Chatterbox TTS backend
```

## Requirements

- Python 3.9+
- ffmpeg (for M4B creation): `brew install ffmpeg` / `apt install ffmpeg`
