#!/usr/bin/env python3
"""
EPUB to MP3 Audiobook Converter

Converts EPUB files to MP3 audiobooks using Modal cloud GPU and Qwen3-TTS.
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import modal

from src.config import (
    DEFAULT_KOKORO_LANGUAGE,
    DEFAULT_KOKORO_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    KOKORO_ALL_VOICES,
    KOKORO_LANG_CODES,
    KOKORO_VOICES,
    LANGUAGES,
    MAX_CHUNKS_PER_BATCH,
    MAX_PARALLEL_CHAPTERS,
    MODEL_CONFIG,
    MODELS,
    PAUSE_BETWEEN_CHUNKS_MS,
    PAUSE_BETWEEN_PARAGRAPHS_MS,
    SPEAKERS,
)
from src.epub_parser import Chapter, get_book_metadata, parse_epub
from src.text_chunker import chunk_text, estimate_audio_duration


def sanitize_filename(name: str) -> str:
    """Create a safe filename from a string."""
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    # Limit length
    return name[:100].strip()


def load_progress(progress_file: Path) -> dict:
    """Load progress from file."""
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"completed_chapters": [], "model": None}


def get_tts_service(model: str):
    """Get the appropriate TTS service for the given model."""
    if model == "chatterbox":
        cls_name = "ChatterboxTTSService"
    elif model == "kokoro":
        cls_name = "KokoroTTSService"
    else:
        cls_name = "TTSService"
    return modal.Cls.from_name("audiobook-tts", cls_name)()


def save_progress(progress_file: Path, progress: dict):
    """Save progress to file."""
    progress_file.write_text(json.dumps(progress, indent=2))


class ThreadSafeProgress:
    """Thread-safe progress tracker for parallel chapter processing."""

    def __init__(self, progress_file: Path, initial_progress: dict):
        self.progress_file = progress_file
        self.progress = initial_progress
        self.lock = threading.Lock()

    def mark_completed(self, chapter_index: int):
        """Mark a chapter as completed (thread-safe)."""
        with self.lock:
            if chapter_index not in self.progress["completed_chapters"]:
                self.progress["completed_chapters"].append(chapter_index)
                save_progress(self.progress_file, self.progress)


def process_chapter(
    chapter: Chapter,
    service,
    speaker: str,
    language: str,
    instruct: str | None,
    output_dir: Path,
    book_title: str,
) -> Path:
    """
    Process a single chapter and save as MP3.

    Args:
        chapter: Chapter to process
        service: Modal TTS service
        speaker: Voice to use
        language: Language for TTS
        instruct: Optional style instruction
        output_dir: Directory to save MP3
        book_title: Book title for filename

    Returns:
        Path to the saved MP3 file
    """
    from pydub import AudioSegment
    import io

    # Chunk the chapter text
    chunks = chunk_text(chapter.content)

    if not chunks:
        print(f"  Warning: No content to synthesize for chapter {chapter.index}")
        return None

    # Prepare data for Modal
    chunk_texts = [c.text for c in chunks]
    paragraph_ends = [c.is_paragraph_end for c in chunks]

    # Estimate duration
    duration_mins = estimate_audio_duration(chunks)
    print(f"  Chunks: {len(chunks)}, Est. duration: {duration_mins:.1f} min")

    # Split into batches if needed
    num_batches = (len(chunks) + MAX_CHUNKS_PER_BATCH - 1) // MAX_CHUNKS_PER_BATCH

    if num_batches == 1:
        # Single batch - process normally
        print(f"  Synthesizing...")
        mp3_bytes = service.synthesize_chapter.remote(
            chunks=chunk_texts,
            paragraph_ends=paragraph_ends,
            speaker=speaker,
            language=language,
            instruct=instruct,
            pause_between_chunks_ms=PAUSE_BETWEEN_CHUNKS_MS,
            pause_between_paragraphs_ms=PAUSE_BETWEEN_PARAGRAPHS_MS,
        )
    else:
        # Multiple batches - process in parallel using .starmap()
        print(f"  Synthesizing in {num_batches} batches (parallel)...")

        # Prepare batch arguments
        batch_args = []
        for batch_idx in range(num_batches):
            start = batch_idx * MAX_CHUNKS_PER_BATCH
            end = min(start + MAX_CHUNKS_PER_BATCH, len(chunks))
            batch_texts = chunk_texts[start:end]
            batch_paragraph_ends = paragraph_ends[start:end]

            batch_args.append((
                batch_texts,
                batch_paragraph_ends,
                speaker,
                language,
                instruct,
                PAUSE_BETWEEN_CHUNKS_MS,
                PAUSE_BETWEEN_PARAGRAPHS_MS,
            ))

        # Process all batches in parallel
        batch_results = list(service.synthesize_chapter.starmap(
            batch_args,
            order_outputs=True,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        ))

        # Check for exceptions and convert to AudioSegments
        mp3_segments = []
        for batch_idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                raise RuntimeError(f"Batch {batch_idx + 1} failed: {result}")
            segment = AudioSegment.from_mp3(io.BytesIO(result))
            mp3_segments.append(segment)

        # Concatenate all segments
        print(f"  Concatenating {num_batches} batches...")
        combined = mp3_segments[0]
        for segment in mp3_segments[1:]:
            # Add inter-chunk pause between batches
            silence = AudioSegment.silent(duration=PAUSE_BETWEEN_CHUNKS_MS)
            combined += silence + segment

        # Export to MP3
        buffer = io.BytesIO()
        combined.export(buffer, format="mp3", bitrate="192k")
        buffer.seek(0)
        mp3_bytes = buffer.read()

    # Save MP3
    chapter_title = sanitize_filename(chapter.title)
    filename = f"{chapter.index:03d}_{chapter_title}.mp3"
    output_path = output_dir / filename
    output_path.write_bytes(mp3_bytes)

    print(f"  Saved: {output_path.name} ({len(mp3_bytes) / 1024 / 1024:.1f} MB)")

    return output_path


def process_chapters_parallel(
    chapters: list[Chapter],
    completed: set[int],
    service,
    speaker: str,
    language: str,
    instruct: str | None,
    output_dir: Path,
    book_title: str,
    progress_tracker: ThreadSafeProgress,
    max_workers: int = MAX_PARALLEL_CHAPTERS,
) -> tuple[int, int]:
    """
    Process multiple chapters in parallel using Modal's native parallelism.

    This uses Modal's .starmap() for true server-side parallelism, allowing
    containers to scale up automatically based on workload.

    Args:
        chapters: List of chapters to process
        completed: Set of already completed chapter indices
        service: Modal TTS service
        speaker: Voice to use
        language: Language for TTS
        instruct: Optional style instruction
        output_dir: Directory to save MP3
        book_title: Book title for filename
        progress_tracker: Thread-safe progress tracker
        max_workers: Maximum number of concurrent chapters

    Returns:
        Tuple of (successful_count, failed_count)
    """
    from pydub import AudioSegment
    import io

    # Filter out already completed chapters
    pending_chapters = [ch for ch in chapters if ch.index not in completed]
    skipped_count = len(chapters) - len(pending_chapters)

    if skipped_count > 0:
        print(f"Skipping {skipped_count} already completed chapters")

    if not pending_chapters:
        return skipped_count, 0

    # Prepare all chapter data for Modal's starmap
    print(f"🚀 Preparing {len(pending_chapters)} chapters for parallel processing...")
    
    chapter_args = []
    chapter_metadata = []  # Track chapter info for saving results
    
    for chapter in pending_chapters:
        chunks = chunk_text(chapter.content)
        
        if not chunks:
            print(f"  ⚠️ Chapter {chapter.index}: No content to synthesize")
            continue
        
        chunk_texts = [c.text for c in chunks]
        paragraph_ends = [c.is_paragraph_end for c in chunks]
        
        # Estimate duration for logging
        duration_mins = estimate_audio_duration(chunks)
        print(f"  Chapter {chapter.index}: {len(chunks)} chunks, ~{duration_mins:.1f} min")
        
        chapter_args.append((
            chunk_texts,
            paragraph_ends,
            speaker,
            language,
            instruct,
            PAUSE_BETWEEN_CHUNKS_MS,
            PAUSE_BETWEEN_PARAGRAPHS_MS,
        ))
        chapter_metadata.append({
            "chapter": chapter,
            "num_chunks": len(chunks),
        })
    
    if not chapter_args:
        return skipped_count, 0
    
    # Fan out to Modal - all chapters process in parallel!
    print(f"\n🔥 Synthesizing {len(chapter_args)} chapters in parallel (up to {max_workers} concurrent)...")
    
    # Use Modal's starmap for server-side parallelism
    results = list(service.synthesize_chapter.starmap(
        chapter_args,
        order_outputs=True,
        return_exceptions=True,
    ))
    
    # Process results and save files
    successful = skipped_count
    failed = 0
    
    for meta, result in zip(chapter_metadata, results):
        chapter = meta["chapter"]
        chapter_title = sanitize_filename(chapter.title)
        filename = f"{chapter.index:03d}_{chapter_title}.mp3"
        output_path = output_dir / filename
        
        if isinstance(result, Exception):
            print(f"  ❌ Chapter {chapter.index}: Failed - {result}")
            failed += 1
        else:
            # Save the MP3
            output_path.write_bytes(result)
            progress_tracker.mark_completed(chapter.index)
            successful += 1
            print(f"  ✅ Chapter {chapter.index}: {filename} ({len(result) / 1024 / 1024:.1f} MB)")
    
    return successful, failed


def convert_epub_to_audiobook(
    epub_path: Path,
    output_dir: Path,
    speaker: str,
    language: str,
    instruct: str | None,
    model: str = DEFAULT_MODEL,
    resume: bool = True,
):
    """
    Convert an EPUB file to MP3 audiobook.

    Args:
        epub_path: Path to EPUB file
        output_dir: Directory to save MP3 files
        speaker: Voice to use
        language: Language for TTS
        instruct: Optional style instruction
        model: TTS model to use (qwen or chatterbox)
        resume: Whether to resume from previous progress
    """
    print(f"\n{'=' * 60}")
    print("EPUB to MP3 Audiobook Converter")
    print(f"{'=' * 60}\n")

    # Validate EPUB
    if not epub_path.exists():
        print(f"Error: EPUB file not found: {epub_path}")
        sys.exit(1)

    # Get book metadata
    metadata = get_book_metadata(epub_path)
    book_title = metadata.get("title") or epub_path.stem
    author = metadata.get("author") or "Unknown"

    print(f"Book: {book_title}")
    print(f"Author: {author}")
    print(f"Model: {model}")
    if model == "qwen":
        print(f"Speaker: {speaker}")
        print(f"Language: {language}")
        if instruct:
            print(f"Style: {instruct}")
    elif model == "kokoro":
        print(f"Voice: {speaker}")
        print(f"Language: {language}")
    else:
        print("Speaker: (default Chatterbox voice)")
        print("Language: English")

    # Convert language to code for Kokoro
    effective_language = language
    if model == "kokoro":
        effective_language = KOKORO_LANG_CODES.get(language, 'a')

    # Parse EPUB
    print("\nParsing EPUB...")
    chapters = parse_epub(epub_path)
    print(f"Found {len(chapters)} chapters")

    if not chapters:
        print("Error: No chapters found in EPUB")
        sys.exit(1)

    # Create output directory
    output_dir = output_dir / sanitize_filename(book_title)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load progress
    progress_file = output_dir / ".progress.json"
    progress = load_progress(progress_file) if resume else {"completed_chapters": [], "model": None}
    completed = set(progress["completed_chapters"])

    # Check for model mismatch
    if resume and progress.get("model") and progress["model"] != model:
        print(f"\nWarning: Previous conversion used model '{progress['model']}', but you specified '{model}'.")
        print("Use --no-resume to start fresh with the new model, or continue with the original model.")
        sys.exit(1)

    if completed:
        print(f"Resuming: {len(completed)} chapters already completed")

    # Save model in progress
    progress["model"] = model
    save_progress(progress_file, progress)

    # Initialize Modal service
    print(f"\nConnecting to Modal TTS service ({model})...")
    service = get_tts_service(model)

    # Process chapters in parallel
    print(f"\n{'=' * 60}")
    print(f"Processing Chapters (max {MAX_PARALLEL_CHAPTERS} parallel)")
    print(f"{'=' * 60}\n")

    # Create thread-safe progress tracker
    progress_tracker = ThreadSafeProgress(progress_file, progress)

    successful, failed = process_chapters_parallel(
        chapters=chapters,
        completed=completed,
        service=service,
        speaker=speaker,
        language=effective_language,
        instruct=instruct,
        output_dir=output_dir,
        book_title=book_title,
        progress_tracker=progress_tracker,
        max_workers=MAX_PARALLEL_CHAPTERS,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Conversion Complete")
    print(f"{'=' * 60}")
    print(f"Successful: {successful}/{len(chapters)}")
    print(f"Failed: {failed}/{len(chapters)}")
    print(f"Output: {output_dir}")

    # Clean up progress file if complete
    if successful == len(chapters) and progress_file.exists():
        progress_file.unlink()
        print("Progress file cleaned up")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert EPUB files to MP3 audiobooks using Kokoro, Qwen3-TTS, or Chatterbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py book.epub                                    # Kokoro (default)
  python main.py book.epub -s bf_alice -l "British English"   # Kokoro with British voice
  python main.py book.epub --model qwen -s Ryan               # Qwen with Ryan voice
  python main.py book.epub --model qwen -s Vivian -l Chinese  # Qwen with Chinese
  python main.py book.epub --model qwen --instruct "Read calmly"
  python main.py book.epub --model chatterbox

Available models: """ + ", ".join(MODELS) + " (default: " + DEFAULT_MODEL + ")"
    )

    parser.add_argument(
        "epub",
        type=Path,
        nargs="?",
        help="Path to the EPUB file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./audiobooks"),
        help="Output directory (default: ./audiobooks)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=MODELS,
        help=f"TTS model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-s", "--speaker",
        type=str,
        default=None,
        help="Voice to use (model-specific, see --list-speakers)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Language for TTS (model-specific, see --list-speakers)"
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for Qwen (e.g., 'Read with enthusiasm')"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any previous progress"
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available speakers and exit"
    )

    args = parser.parse_args()

    if args.list_speakers:
        model_config = MODEL_CONFIG[args.model]
        if args.model == "kokoro":
            print("Kokoro-82M voices by language:\n")
            for lang, voices in KOKORO_VOICES.items():
                print(f"  {lang}:")
                for voice in voices:
                    default_marker = " (default)" if voice == DEFAULT_KOKORO_VOICE else ""
                    print(f"    - {voice}{default_marker}")
                print()
            print(f"Default language: {DEFAULT_KOKORO_LANGUAGE}")
        elif model_config["speakers"]:
            print(f"Available speakers for {args.model}:")
            for speaker in model_config["speakers"]:
                print(f"  - {speaker}")
            print(f"\nSupported languages: {', '.join(model_config['languages'])}")
        else:
            print(f"Model '{args.model}' uses a default voice (no speaker selection).")
            print(f"Supported languages: {', '.join(model_config['languages'])}")
        print(f"Supports style instructions: {'Yes' if model_config['supports_instruct'] else 'No'}")
        sys.exit(0)

    if args.epub is None:
        parser.error("the following arguments are required: epub")

    # Apply model-specific defaults and validation
    if args.model == "kokoro":
        args.speaker = args.speaker or DEFAULT_KOKORO_VOICE
        args.language = args.language or DEFAULT_KOKORO_LANGUAGE
        if args.speaker not in KOKORO_ALL_VOICES:
            parser.error(f"Invalid voice '{args.speaker}'. Use --list-speakers to see available voices.")
        if args.language not in KOKORO_LANG_CODES:
            parser.error(f"Invalid language '{args.language}'. Choose from: {', '.join(KOKORO_LANG_CODES.keys())}")
        if args.instruct:
            print("Warning: --instruct is ignored for Kokoro (no style instructions)")
    elif args.model == "chatterbox":
        args.speaker = args.speaker or ""
        args.language = args.language or "English"
        # Warn about ignored options
        if args.speaker:
            print("Warning: --speaker is ignored for Chatterbox (uses default voice)")
        if args.language != "English":
            print("Warning: --language is ignored for Chatterbox (English only)")
        if args.instruct:
            print("Warning: --instruct is ignored for Chatterbox (no style instructions)")
    else:
        # Qwen defaults and validation
        args.speaker = args.speaker or DEFAULT_SPEAKER
        args.language = args.language or DEFAULT_LANGUAGE
        if args.speaker not in SPEAKERS:
            parser.error(f"Invalid speaker '{args.speaker}'. Choose from: {', '.join(SPEAKERS)}")
        if args.language not in LANGUAGES:
            parser.error(f"Invalid language '{args.language}'. Choose from: {', '.join(LANGUAGES)}")

    convert_epub_to_audiobook(
        epub_path=args.epub,
        output_dir=args.output,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct,
        model=args.model,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
