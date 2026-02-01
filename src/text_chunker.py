"""Text chunking utilities for TTS processing."""

import re
from dataclasses import dataclass

from .config import MAX_CHUNK_LENGTH, MIN_CHUNK_LENGTH


@dataclass
class TextChunk:
    """A chunk of text ready for TTS synthesis."""

    index: int
    text: str
    is_paragraph_end: bool = False

    def __repr__(self) -> str:
        return f"TextChunk({self.index}, {len(self.text)} chars)"


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common abbreviations to avoid false splits
    text = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|viz|al|eg|ie)\.", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)  # Numbers with periods

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Restore dots
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    return [s.strip() for s in sentences if s.strip()]


def split_long_sentence(sentence: str, max_length: int) -> list[str]:
    """Split a long sentence at clause boundaries."""
    if len(sentence) <= max_length:
        return [sentence]

    parts = []

    # Try splitting at clause boundaries (commas, semicolons, colons, dashes)
    clause_pattern = r"(?<=[,;:\-\u2013\u2014])\s+"
    clauses = re.split(clause_pattern, sentence)

    current_part = ""
    for clause in clauses:
        if len(current_part) + len(clause) + 1 <= max_length:
            current_part = f"{current_part} {clause}".strip() if current_part else clause
        else:
            if current_part:
                parts.append(current_part)

            # If single clause is still too long, split at word boundaries
            if len(clause) > max_length:
                words = clause.split()
                current_part = ""
                for word in words:
                    if len(current_part) + len(word) + 1 <= max_length:
                        current_part = f"{current_part} {word}".strip() if current_part else word
                    else:
                        if current_part:
                            parts.append(current_part)
                        current_part = word
            else:
                current_part = clause

    if current_part:
        parts.append(current_part)

    return parts


def chunk_text(text: str, max_length: int = MAX_CHUNK_LENGTH) -> list[TextChunk]:
    """
    Split text into chunks suitable for TTS synthesis.

    Args:
        text: The text to chunk
        max_length: Maximum characters per chunk

    Returns:
        List of TextChunk objects
    """
    if not text or not text.strip():
        return []

    chunks = []
    chunk_index = 0

    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    current_chunk = ""

    for para_idx, paragraph in enumerate(paragraphs):
        is_last_paragraph = para_idx == len(paragraphs) - 1
        sentences = split_into_sentences(paragraph)

        for sent_idx, sentence in enumerate(sentences):
            is_last_sentence = sent_idx == len(sentences) - 1

            # Handle sentences that are too long
            if len(sentence) > max_length:
                # Save current chunk first
                if current_chunk:
                    chunks.append(TextChunk(
                        index=chunk_index,
                        text=current_chunk,
                        is_paragraph_end=False
                    ))
                    chunk_index += 1
                    current_chunk = ""

                # Split and add long sentence parts
                parts = split_long_sentence(sentence, max_length)
                for i, part in enumerate(parts):
                    is_last_part = i == len(parts) - 1
                    chunks.append(TextChunk(
                        index=chunk_index,
                        text=part,
                        is_paragraph_end=is_last_part and is_last_sentence and not is_last_paragraph
                    ))
                    chunk_index += 1
                continue

            # Check if adding sentence would exceed limit
            potential_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

            if len(potential_chunk) <= max_length:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(TextChunk(
                        index=chunk_index,
                        text=current_chunk,
                        is_paragraph_end=False
                    ))
                    chunk_index += 1
                current_chunk = sentence

        # End of paragraph - mark and potentially save chunk
        if current_chunk and len(current_chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(TextChunk(
                index=chunk_index,
                text=current_chunk,
                is_paragraph_end=not is_last_paragraph
            ))
            chunk_index += 1
            current_chunk = ""

    # Save any remaining text
    if current_chunk:
        chunks.append(TextChunk(
            index=chunk_index,
            text=current_chunk,
            is_paragraph_end=False
        ))

    return chunks


def estimate_audio_duration(chunks: list[TextChunk], words_per_minute: int = 150) -> float:
    """
    Estimate audio duration in minutes.

    Args:
        chunks: List of text chunks
        words_per_minute: Average speaking rate

    Returns:
        Estimated duration in minutes
    """
    total_words = sum(len(chunk.text.split()) for chunk in chunks)
    return total_words / words_per_minute
