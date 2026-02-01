#!/usr/bin/env python3
"""
Batch EPUB to Audiobook Converter

Processes all EPUB files in a directory, skipping non-EPUB files.
Tracks progress across runs and moves completed books.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_epub_files(input_dir: Path) -> list[Path]:
    """Get all EPUB files in directory, sorted by size (smallest first)."""
    epubs = list(input_dir.glob("*.epub"))
    return sorted(epubs, key=lambda p: p.stat().st_size)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert EPUB files to audiobooks"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing EPUB files"
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
        default="kokoro",
        choices=["kokoro", "qwen", "chatterbox"],
        help="TTS model to use (default: kokoro)"
    )
    parser.add_argument(
        "-s", "--speaker",
        type=str,
        default=None,
        help="Voice to use (model-specific)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Language for TTS"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without processing"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}")
        sys.exit(1)

    # Get EPUB files
    epubs = get_epub_files(args.input_dir)
    
    if not epubs:
        print(f"No EPUB files found in {args.input_dir}")
        sys.exit(0)

    # Calculate total size
    total_size_mb = sum(p.stat().st_size for p in epubs) / 1024 / 1024
    
    print(f"\n{'=' * 60}")
    print(f"Batch EPUB to Audiobook Converter")
    print(f"{'=' * 60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Total books: {len(epubs)}")
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("Books to process (sorted by size):\n")
        for i, epub in enumerate(epubs, 1):
            size_mb = epub.stat().st_size / 1024 / 1024
            print(f"  {i:3}. [{size_mb:6.1f} MB] {epub.name}")
        print(f"\nTotal: {len(epubs)} books, {total_size_mb:.1f} MB")
        sys.exit(0)

    # Process each book
    successful = 0
    failed = 0
    
    for i, epub in enumerate(epubs, 1):
        size_mb = epub.stat().st_size / 1024 / 1024
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(epubs)}] Processing: {epub.name}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"{'=' * 60}\n")

        # Build command
        cmd = [
            sys.executable, "main.py",
            str(epub),
            "-o", str(args.output),
            "-m", args.model,
        ]
        
        if args.speaker:
            cmd.extend(["-s", args.speaker])
        if args.language:
            cmd.extend(["-l", args.language])

        try:
            result = subprocess.run(cmd, check=True)
            successful += 1
            print(f"\n✅ Completed: {epub.name}")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"\n❌ Failed: {epub.name} (exit code {e.returncode})")
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Interrupted by user")
            print(f"Progress: {successful} successful, {failed} failed, {len(epubs) - i} remaining")
            sys.exit(1)

    # Summary
    print(f"\n{'=' * 60}")
    print("Batch Conversion Complete")
    print(f"{'=' * 60}")
    print(f"✅ Successful: {successful}/{len(epubs)}")
    print(f"❌ Failed: {failed}/{len(epubs)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
