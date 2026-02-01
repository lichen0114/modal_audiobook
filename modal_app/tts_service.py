"""Modal TTS Service using Qwen3-TTS and Chatterbox."""

import io
import tempfile
from pathlib import Path

import modal


def create_qwen_image() -> modal.Image:
    """Create Modal image with Qwen3-TTS dependencies."""
    return (
        # Use CUDA devel image for nvcc (required to compile flash-attn)
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04",
            add_python="3.12",
        )
        # Install system dependencies
        .apt_install("ffmpeg", "git", "sox", "libsox-dev")
        # Install PyTorch with CUDA support
        .pip_install(
            "torch==2.6.0",
            "torchaudio==2.6.0",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        # Install build dependencies first (needed for flash-attn build)
        .pip_install("packaging", "ninja", "numpy", "wheel", "setuptools")
        # Install flash-attn
        .pip_install(
            "flash-attn",
            extra_options="--no-build-isolation",
        )
        # Install qwen-tts and audio dependencies
        .pip_install(
            "qwen-tts>=0.0.5",
            "soundfile>=0.13.0",
            "pydub>=0.25.0",
            "transformers>=4.40.0",
            "accelerate>=0.30.0",
        )
    )


def create_chatterbox_image() -> modal.Image:
    """Create Modal image with Chatterbox TTS dependencies."""
    return (
        # Use Python 3.11 for chatterbox-tts compatibility (numpy version constraints)
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("ffmpeg", "git", "sox", "libsox-dev")
        .pip_install(
            "torch==2.5.1",
            "torchaudio==2.5.1",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            "chatterbox-tts",
            "soundfile>=0.13.0",
            "pydub>=0.25.0",
        )
    )


def create_kokoro_image() -> modal.Image:
    """Create Modal image with Kokoro-82M TTS dependencies."""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("ffmpeg", "espeak-ng")  # espeak-ng required for Kokoro
        .pip_install(
            "torch==2.5.1",
            "torchaudio==2.5.1",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            "kokoro>=0.9.2",
            "soundfile>=0.13.0",
            "pydub>=0.25.0",
        )
    )


# Pre-built images for faster cold starts
qwen_image = create_qwen_image()
chatterbox_image = create_chatterbox_image()
kokoro_image = create_kokoro_image()

# Create Modal app
app = modal.App("audiobook-tts")

# Volume for caching model weights
model_cache = modal.Volume.from_name("audiobook-model-cache", create_if_missing=True)

MODEL_CACHE_PATH = "/cache/models"
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


@app.cls(
    image=qwen_image,
    gpu="A10G",
    timeout=1800,  # 30 minutes for large chapters
    volumes={MODEL_CACHE_PATH: model_cache},
    scaledown_window=120,  # Auto-stop after 2 min idle
    max_containers=5,  # Limit concurrent containers
)
class TTSService:
    """Text-to-Speech service using Qwen3-TTS on Modal."""

    @modal.enter()
    def load_model(self):
        """Load the TTS model when container starts."""
        import os
        import torch
        from qwen_tts import Qwen3TTSModel

        # Set cache directory for transformers
        os.environ["HF_HOME"] = MODEL_CACHE_PATH
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH

        print(f"Loading model {MODEL_NAME}...")
        self.tts = Qwen3TTSModel.from_pretrained(
            MODEL_NAME,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Use PyTorch's native SDPA instead of flash_attn
        )
        print("Model loaded successfully!")

        # Warmup inference
        print("Running warmup inference...")
        wavs, sr = self.tts.generate_custom_voice(
            text="Hello, this is a warmup.",
            speaker="Ryan",
            language="English",
        )
        print("Warmup complete!")

    @modal.method()
    def synthesize_chunk(
        self,
        text: str,
        speaker: str = "Ryan",
        language: str = "English",
        instruct: str | None = None,
    ) -> bytes:
        """
        Synthesize a single text chunk to WAV audio.

        Args:
            text: Text to synthesize
            speaker: Voice to use
            language: Language of the text
            instruct: Optional style instruction

        Returns:
            WAV audio bytes
        """
        import soundfile as sf
        import numpy as np

        # Synthesize
        audio_array, sample_rate = self.tts.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct if instruct else "",
        )
        audio_array = audio_array[0]  # generate_custom_voice returns list

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)

        return buffer.read()

    @modal.method()
    def synthesize_chapter(
        self,
        chunks: list[str],
        paragraph_ends: list[bool],
        speaker: str = "Ryan",
        language: str = "English",
        instruct: str | None = None,
        pause_between_chunks_ms: int = 500,
        pause_between_paragraphs_ms: int = 800,
    ) -> bytes:
        """
        Synthesize multiple chunks and concatenate with pauses.

        Args:
            chunks: List of text chunks
            paragraph_ends: Boolean flags indicating paragraph ends
            speaker: Voice to use
            language: Language of the text
            instruct: Optional style instruction
            pause_between_chunks_ms: Pause duration between chunks
            pause_between_paragraphs_ms: Pause duration between paragraphs

        Returns:
            MP3 audio bytes
        """
        import numpy as np
        import soundfile as sf
        from pydub import AudioSegment

        if not chunks:
            raise ValueError("No chunks provided")

        audio_segments = []
        sample_rate = None

        for i, (chunk_text, is_paragraph_end) in enumerate(zip(chunks, paragraph_ends)):
            print(f"Synthesizing chunk {i + 1}/{len(chunks)}...")

            # Synthesize
            audio_array, sr = self.tts.generate_custom_voice(
                text=chunk_text,
                speaker=speaker,
                language=language,
                instruct=instruct if instruct else "",
            )
            audio_array = audio_array[0]  # generate_custom_voice returns list

            if sample_rate is None:
                sample_rate = sr

            # Convert numpy array to AudioSegment
            # Normalize to 16-bit PCM
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767).astype(np.int16)

            segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
                channels=1
            )
            audio_segments.append(segment)

            # Add appropriate pause
            if i < len(chunks) - 1:  # Not the last chunk
                pause_ms = pause_between_paragraphs_ms if is_paragraph_end else pause_between_chunks_ms
                silence = AudioSegment.silent(duration=pause_ms)
                audio_segments.append(silence)

        # Concatenate all segments
        print("Concatenating audio segments...")
        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined += segment

        # Export to MP3
        print("Exporting to MP3...")
        buffer = io.BytesIO()
        combined.export(buffer, format="mp3", bitrate="192k")
        buffer.seek(0)

        return buffer.read()

    @modal.method()
    def get_available_speakers(self) -> list[str]:
        """Return list of available speakers."""
        return [
            "Chelsie",
            "Ethan",
            "Aiden",
            "Ryan",
            "Emily",
            "Vivian",
            "Bella",
            "Serena",
            "Aurora",
        ]


@app.cls(
    image=chatterbox_image,
    gpu="A10G",
    timeout=1800,  # 30 minutes for large chapters
    volumes={MODEL_CACHE_PATH: model_cache},
    scaledown_window=120,  # Auto-stop after 2 min idle
    max_containers=5,  # Limit concurrent containers
)
class ChatterboxTTSService:
    """Text-to-Speech service using Chatterbox on Modal."""

    @modal.enter()
    def load_model(self):
        """Load the Chatterbox model when container starts."""
        import os
        from chatterbox.tts import ChatterboxTTS

        os.environ["HF_HOME"] = MODEL_CACHE_PATH
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH

        print("Loading Chatterbox model...")
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        print("Chatterbox model loaded successfully!")

        # Warmup inference
        print("Running warmup inference...")
        _ = self.model.generate("Hello, this is a warmup.")
        print("Warmup complete!")

    @modal.method()
    def synthesize_chunk(
        self,
        text: str,
        speaker: str = "",
        language: str = "English",
        instruct: str | None = None,
    ) -> bytes:
        """
        Synthesize a single text chunk to WAV audio.

        Args:
            text: Text to synthesize
            speaker: Ignored for Chatterbox (no speaker selection)
            language: Ignored for Chatterbox (English only)
            instruct: Ignored for Chatterbox (no style instructions)

        Returns:
            WAV audio bytes
        """
        import soundfile as sf
        import torchaudio

        # Generate audio
        wav = self.model.generate(text)
        sample_rate = self.model.sr

        # Convert tensor to numpy
        audio_array = wav.squeeze().cpu().numpy()

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)

        return buffer.read()

    @modal.method()
    def synthesize_chapter(
        self,
        chunks: list[str],
        paragraph_ends: list[bool],
        speaker: str = "",
        language: str = "English",
        instruct: str | None = None,
        pause_between_chunks_ms: int = 500,
        pause_between_paragraphs_ms: int = 800,
    ) -> bytes:
        """
        Synthesize multiple chunks and concatenate with pauses.

        Args:
            chunks: List of text chunks
            paragraph_ends: Boolean flags indicating paragraph ends
            speaker: Ignored for Chatterbox (no speaker selection)
            language: Ignored for Chatterbox (English only)
            instruct: Ignored for Chatterbox (no style instructions)
            pause_between_chunks_ms: Pause duration between chunks
            pause_between_paragraphs_ms: Pause duration between paragraphs

        Returns:
            MP3 audio bytes
        """
        import numpy as np
        import soundfile as sf
        from pydub import AudioSegment

        if not chunks:
            raise ValueError("No chunks provided")

        audio_segments = []
        sample_rate = self.model.sr

        for i, (chunk_text, is_paragraph_end) in enumerate(zip(chunks, paragraph_ends)):
            print(f"Synthesizing chunk {i + 1}/{len(chunks)}...")

            # Generate audio
            wav = self.model.generate(chunk_text)
            audio_array = wav.squeeze().cpu().numpy()

            # Convert numpy array to AudioSegment
            # Normalize to 16-bit PCM
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767).astype(np.int16)

            segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
                channels=1
            )
            audio_segments.append(segment)

            # Add appropriate pause
            if i < len(chunks) - 1:  # Not the last chunk
                pause_ms = pause_between_paragraphs_ms if is_paragraph_end else pause_between_chunks_ms
                silence = AudioSegment.silent(duration=pause_ms)
                audio_segments.append(silence)

        # Concatenate all segments
        print("Concatenating audio segments...")
        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined += segment

        # Export to MP3
        print("Exporting to MP3...")
        buffer = io.BytesIO()
        combined.export(buffer, format="mp3", bitrate="192k")
        buffer.seek(0)

        return buffer.read()

    @modal.method()
    def get_available_speakers(self) -> list[str]:
        """Return list of available speakers (empty for Chatterbox)."""
        return []


@app.cls(
    image=kokoro_image,
    gpu="T4",  # T4 is sufficient for the 82M model and 46% cheaper than A10G
    timeout=1800,  # 30 minutes for large chapters
    volumes={MODEL_CACHE_PATH: model_cache},
    scaledown_window=120,  # Auto-stop after 2 min idle
    max_containers=8,  # More containers since T4 is cheaper
)
class KokoroTTSService:
    """Text-to-Speech service using Kokoro-82M on Modal."""

    @modal.enter()
    def load_model(self):
        """Load the Kokoro model when container starts."""
        import os
        from kokoro import KPipeline

        os.environ["HF_HOME"] = MODEL_CACHE_PATH
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH

        print("Loading Kokoro-82M model...")
        # Initialize with American English by default
        self.pipeline = KPipeline(lang_code='a')
        self.current_lang_code = 'a'
        self.sample_rate = 24000
        print("Kokoro model loaded successfully!")

        # Warmup inference
        print("Running warmup inference...")
        for gs, ps, audio in self.pipeline("Hello, this is a warmup.", voice='af_heart'):
            pass  # Just run through the generator
        print("Warmup complete!")

    def _ensure_language(self, lang_code: str):
        """Switch pipeline language if needed."""
        if lang_code != self.current_lang_code:
            from kokoro import KPipeline
            print(f"Switching language from '{self.current_lang_code}' to '{lang_code}'...")
            self.pipeline = KPipeline(lang_code=lang_code)
            self.current_lang_code = lang_code

    @modal.method()
    def synthesize_chunk(
        self,
        text: str,
        speaker: str = "af_heart",
        language: str = "a",
        instruct: str | None = None,
    ) -> bytes:
        """
        Synthesize a single text chunk to WAV audio.

        Args:
            text: Text to synthesize
            speaker: Kokoro voice to use (e.g., 'af_heart')
            language: Language code (e.g., 'a' for American English)
            instruct: Ignored for Kokoro (no style instructions)

        Returns:
            WAV audio bytes
        """
        import numpy as np
        import soundfile as sf

        self._ensure_language(language)

        # Collect audio from generator
        audio_parts = []
        for gs, ps, audio in self.pipeline(text, voice=speaker):
            audio_parts.append(audio)

        if not audio_parts:
            # Return empty audio if nothing generated
            audio_array = np.zeros(0, dtype=np.float32)
        else:
            audio_array = np.concatenate(audio_parts)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, self.sample_rate, format="WAV")
        buffer.seek(0)

        return buffer.read()

    @modal.method()
    def synthesize_chapter(
        self,
        chunks: list[str],
        paragraph_ends: list[bool],
        speaker: str = "af_heart",
        language: str = "a",
        instruct: str | None = None,
        pause_between_chunks_ms: int = 500,
        pause_between_paragraphs_ms: int = 800,
    ) -> bytes:
        """
        Synthesize multiple chunks and concatenate with pauses.

        Args:
            chunks: List of text chunks
            paragraph_ends: Boolean flags indicating paragraph ends
            speaker: Kokoro voice to use
            language: Language code (e.g., 'a' for American English)
            instruct: Ignored for Kokoro (no style instructions)
            pause_between_chunks_ms: Pause duration between chunks
            pause_between_paragraphs_ms: Pause duration between paragraphs

        Returns:
            MP3 audio bytes
        """
        import numpy as np
        from pydub import AudioSegment

        if not chunks:
            raise ValueError("No chunks provided")

        self._ensure_language(language)

        audio_segments = []

        for i, (chunk_text, is_paragraph_end) in enumerate(zip(chunks, paragraph_ends)):
            print(f"Synthesizing chunk {i + 1}/{len(chunks)}...")

            # Collect audio from generator
            audio_parts = []
            for gs, ps, audio in self.pipeline(chunk_text, voice=speaker):
                audio_parts.append(audio)

            if audio_parts:
                audio_array = np.concatenate(audio_parts)

                # Convert numpy array to AudioSegment
                # Normalize to 16-bit PCM
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_int16 = (audio_array * 32767).astype(np.int16)

                segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,  # 16-bit
                    channels=1
                )
                audio_segments.append(segment)

                # Add appropriate pause
                if i < len(chunks) - 1:  # Not the last chunk
                    pause_ms = pause_between_paragraphs_ms if is_paragraph_end else pause_between_chunks_ms
                    silence = AudioSegment.silent(duration=pause_ms)
                    audio_segments.append(silence)

        if not audio_segments:
            raise ValueError("No audio generated")

        # Concatenate all segments
        print("Concatenating audio segments...")
        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined += segment

        # Export to MP3
        print("Exporting to MP3...")
        buffer = io.BytesIO()
        combined.export(buffer, format="mp3", bitrate="192k")
        buffer.seek(0)

        return buffer.read()

    @modal.method()
    def get_available_speakers(self) -> list[str]:
        """Return list of available Kokoro voices."""
        return [
            # American English
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck", "am_santa",
            # British English
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            # Japanese
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
            # Chinese
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
            # Spanish
            "ef_dora", "em_alex", "em_santa",
            # French
            "ff_siwis",
            # Hindi
            "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
            # Italian
            "if_sara", "im_nicola",
            # Brazilian Portuguese
            "pf_dora", "pm_alex", "pm_santa",
        ]


# For testing the service directly
@app.local_entrypoint()
def main(model: str = "kokoro"):
    """Test the TTS service."""
    print(f"Testing TTS Service (model: {model})...")

    if model == "chatterbox":
        service = ChatterboxTTSService()
        speaker = ""
        language = "English"
    elif model == "kokoro":
        service = KokoroTTSService()
        speaker = "af_heart"
        language = "a"  # Language code for American English
    else:
        service = TTSService()
        speaker = "Ryan"
        language = "English"

    # Test single chunk
    print("\nTesting single chunk synthesis...")
    audio_bytes = service.synthesize_chunk.remote(
        text="Hello! This is a test of the Text-to-Speech system.",
        speaker=speaker,
        language=language,
    )
    print(f"Generated {len(audio_bytes)} bytes of WAV audio")

    # Test chapter synthesis
    print("\nTesting chapter synthesis...")
    chunks = [
        "This is the first paragraph of our test chapter.",
        "Here is the second paragraph with more content.",
        "And finally, the third paragraph concludes our test.",
    ]
    paragraph_ends = [True, True, False]

    mp3_bytes = service.synthesize_chapter.remote(
        chunks=chunks,
        paragraph_ends=paragraph_ends,
        speaker=speaker,
        language=language,
    )
    print(f"Generated {len(mp3_bytes)} bytes of MP3 audio")

    # Save test output
    output_path = Path(f"test_output_{model}.mp3")
    output_path.write_bytes(mp3_bytes)
    print(f"Saved test audio to {output_path}")

    print("\nTTS Service test complete!")
