"""Modal GPU image configuration for Qwen3-TTS."""

import modal

# Create the Modal image with all dependencies
def create_tts_image() -> modal.Image:
    """Create Modal image with Qwen3-TTS dependencies."""

    return (
        # Use CUDA devel image for nvcc (required to compile flash-attn)
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04",
            add_python="3.12",
        )
        # Install system dependencies
        .apt_install("ffmpeg", "git")
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


# Pre-built image for faster cold starts
tts_image = create_tts_image()
