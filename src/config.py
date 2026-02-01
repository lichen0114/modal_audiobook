"""Configuration constants for the audiobook converter."""

# Kokoro-82M voices by language
KOKORO_VOICES = {
    "American English": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
                         "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
                         "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                         "am_michael", "am_onyx", "am_puck", "am_santa"],
    "British English": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                        "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    "Japanese": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "Chinese": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
    "Spanish": ["ef_dora", "em_alex", "em_santa"],
    "French": ["ff_siwis"],
    "Hindi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "Italian": ["if_sara", "im_nicola"],
    "Brazilian Portuguese": ["pf_dora", "pm_alex", "pm_santa"],
}

KOKORO_ALL_VOICES = [v for voices in KOKORO_VOICES.values() for v in voices]

KOKORO_LANG_CODES = {
    "American English": "a", "British English": "b", "Japanese": "j",
    "Chinese": "z", "Spanish": "e", "French": "f",
    "Hindi": "h", "Italian": "i", "Brazilian Portuguese": "p",
}

DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_KOKORO_LANGUAGE = "American English"

# Available speakers for Qwen3-TTS
SPEAKERS = [
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

# Available languages
LANGUAGES = [
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Default settings
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"

# Model configuration
MODELS = ["kokoro", "qwen", "chatterbox"]
DEFAULT_MODEL = "kokoro"

MODEL_CONFIG = {
    "kokoro": {
        "speakers": KOKORO_ALL_VOICES,
        "languages": list(KOKORO_LANG_CODES.keys()),
        "supports_instruct": False,
    },
    "qwen": {
        "speakers": SPEAKERS,
        "languages": LANGUAGES,
        "supports_instruct": True,
    },
    "chatterbox": {
        "speakers": [],
        "languages": ["English"],
        "supports_instruct": False,
    },
}

# Text chunking settings
MAX_CHUNK_LENGTH = 2000  # Maximum characters per chunk
MIN_CHUNK_LENGTH = 100   # Minimum characters per chunk

# Audio settings
PAUSE_BETWEEN_CHUNKS_MS = 500  # Pause between chunks in milliseconds
PAUSE_BETWEEN_PARAGRAPHS_MS = 800  # Pause between paragraphs

# Batch processing settings
MAX_CHUNKS_PER_BATCH = 30  # Max chunks per Modal call to avoid timeouts

# Parallel processing settings
MAX_PARALLEL_CHAPTERS = 4  # Number of chapters to process concurrently
