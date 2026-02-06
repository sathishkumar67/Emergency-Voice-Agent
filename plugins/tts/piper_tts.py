"""Piper TTS Plugin for LiveKit Agents

Fast local neural TTS using ONNX models.
Native support: Hindi (hi), Telugu (te)
Fallback mapping: Dravidian languages (kn, ta, ml) -> Telugu, Others -> Hindi

Reference: https://github.com/rhasspy/piper
Models: https://huggingface.co/rhasspy/piper-voices
Samples: https://rhasspy.github.io/piper-samples/
"""

import logging
import uuid
import time
import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("piper-tts")

# Lazy import piper to allow graceful handling if not installed
PiperVoice = None
SynthesisConfig = None


def _load_piper():
    global PiperVoice, SynthesisConfig
    if PiperVoice is None:
        try:
            from piper import PiperVoice as PV, SynthesisConfig as SC
            PiperVoice = PV
            SynthesisConfig = SC
        except ImportError:
            raise ImportError("piper-tts not installed. Run: pip install piper-tts")
    return PiperVoice, SynthesisConfig


# Available Indian language voices in Piper
# Format: {voice_id}: {model_name, display_name, lang_code}
PIPER_VOICES = {
    # Hindi voices
    "hi_rohan": {"model": "hi_IN-rohan-medium", "name": "Hindi (Rohan)", "lang": "hi"},
    "hi_pratham": {"model": "hi_IN-pratham-medium", "name": "Hindi (Pratham)", "lang": "hi"},
    "hi_priyamvada": {"model": "hi_IN-priyamvada-medium", "name": "Hindi (Priyamvada)", "lang": "hi"},
    # Telugu
    "te_maya": {"model": "te_IN-maya-medium", "name": "Telugu (Maya)", "lang": "te"},
    # English
    "en_lessac": {"model": "en_US-lessac-medium", "name": "English (Lessac)", "lang": "en"},
    # Nepali
    "ne_chitwan": {"model": "ne_NP-chitwan-medium", "name": "Nepali (Chitwan)", "lang": "ne"},
}

# Language to voice mapping based on linguistic/script similarity
# Dravidian languages (similar scripts): Kannada, Tamil, Malayalam -> Telugu
# Indo-Aryan languages (Devanagari-like): Bengali, Marathi, etc. -> Hindi
LANG_TO_VOICE = {
    # Native support
    "hi": "hi_rohan",
    "te": "te_maya",
    # Dravidian family -> Telugu (similar script structure)
    "kn": "te_maya",   # Kannada -> Telugu (both Dravidian, similar scripts)
    "ta": "te_maya",   # Tamil -> Telugu (Dravidian family)
    "ml": "te_maya",   # Malayalam -> Telugu (Dravidian family)
    # Indo-Aryan family -> Hindi (Devanagari-based)
    "bn": "hi_rohan",  # Bengali -> Hindi
    "mr": "hi_rohan",  # Marathi -> Hindi (same Devanagari script)
    "gu": "hi_rohan",  # Gujarati -> Hindi
    "pa": "hi_rohan",  # Punjabi -> Hindi
    "ne": "hi_rohan",  # Nepali -> Hindi (Devanagari script)
    "or": "hi_rohan",  # Odia -> Hindi
    "as": "hi_rohan",  # Assamese -> Hindi
    "sa": "hi_rohan",  # Sanskrit -> Hindi
    "en": "en_lessac",  # English (native)
}


@dataclass
class PiperOptions:
    models_dir: Path
    voice: str = "hi_rohan"
    use_cuda: bool = False
    length_scale: float = 1.0  # 1.0 = normal speed


class TTS(tts.TTS):
    """Piper TTS - Local neural TTS using ONNX models."""

    def __init__(
        self,
        *,
        models_dir: str = "/home/onprem/tts/piper/models",
        voice: str = "hi_rohan",
        use_cuda: bool = False,
        length_scale: float = 1.0,
    ):
        # Piper outputs 22050 Hz audio
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=22050,
            num_channels=1,
        )
        self._opts = PiperOptions(
            models_dir=Path(models_dir),
            voice=voice,
            use_cuda=use_cuda,
            length_scale=length_scale,
        )
        self._voice_cache: dict[str, any] = {}

    def _get_voice(self, voice_id: str):
        """Load and cache voice model."""
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]

        PV, _ = _load_piper()

        voice_info = PIPER_VOICES.get(voice_id)
        if not voice_info:
            logger.warning(f"Unknown voice '{voice_id}', falling back to hi_rohan")
            voice_id = "hi_rohan"
            voice_info = PIPER_VOICES[voice_id]

        model_name = voice_info["model"]
        model_path = self._opts.models_dir / f"{model_name}.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {model_path}\n"
                f"Download with: python -m piper.download --output-dir {self._opts.models_dir} {model_name}"
            )

        logger.info(f"Loading Piper voice: {voice_id} from {model_path}")
        voice = PV.load(str(model_path), use_cuda=self._opts.use_cuda)
        self._voice_cache[voice_id] = voice
        return voice

    def update_options(
        self,
        *,
        voice: Optional[str] = None,
        length_scale: Optional[float] = None,
    ):
        """Update TTS options dynamically.

        Args:
            voice: Voice ID (e.g., "hi_rohan", "te_maya")
            length_scale: Speech speed (1.0 = normal, 2.0 = half speed)
        """
        if voice:
            self._opts.voice = voice
            logger.info(f"Piper TTS voice updated to: {voice}")
        if length_scale is not None:
            self._opts.length_scale = length_scale

    async def aclose(self):
        self._voice_cache.clear()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "PiperStream":
        return PiperStream(
            tts=self, input_text=text, opts=self._opts, conn_options=conn_options
        )


class PiperStream(tts.ChunkedStream):
    """Synthesizes audio using local Piper model."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: PiperOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter):
        _, SC = _load_piper()

        text_preview = self._input_text[:50].replace('\n', ' ')
        voice_info = PIPER_VOICES.get(self._opts.voice, PIPER_VOICES["hi_rohan"])
        logger.info(f"Piper TTS: '{text_preview}...' [voice={self._opts.voice}]")

        tts_start = time.time()

        # Get the voice model
        voice = self._tts._get_voice(self._opts.voice)

        # Configure synthesis
        syn_config = SC(length_scale=self._opts.length_scale)

        # Synthesize to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(self._input_text, wav_file, syn_config=syn_config)

        # Extract PCM data from WAV
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            pcm_data = wav_file.readframes(wav_file.getnframes())

        tts_end = time.time()
        duration = len(pcm_data) / 2 / sample_rate  # 16-bit = 2 bytes per sample
        logger.info(
            f"[TIMING] Piper TTS: {(tts_end - tts_start)*1000:.0f}ms "
            f"({len(self._input_text)} chars -> {duration:.2f}s audio)"
        )

        # Emit audio (framework calls end_input() automatically after _run completes)
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm_data)


def get_voice_for_lang(lang_code: str) -> str:
    """Get best voice for a language code.

    Args:
        lang_code: ISO language code (hi, te, kn, ta, ml, etc.)

    Returns:
        Voice ID like "hi_rohan" or "te_maya"
    """
    voice_id = LANG_TO_VOICE.get(lang_code, "hi_rohan")

    # Log mapping info for non-native languages
    if lang_code not in ["hi", "te"]:
        mapped_lang = "Telugu" if voice_id == "te_maya" else "Hindi"
        logger.info(f"Language '{lang_code}' mapped to {mapped_lang} voice ({voice_id})")

    return voice_id
