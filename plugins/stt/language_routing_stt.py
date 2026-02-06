"""Language Routing STT - Routes to AIBharath for Indian languages, Whisper otherwise.

Uses VoxLingua107 for language detection with majority vote.
Supports auto detection or fixed language mode.
"""

import io
import os
import wave
import time
import logging
from collections import Counter
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from .whisper_stt import STT as WhisperSTT
from .aibharath_conformer_stt import STT as AIBharathSTT, SUPPORTED_LANGUAGES

logger = logging.getLogger("language-routing-stt")

INDIAN_LANGUAGES = set(SUPPORTED_LANGUAGES)
MIN_DETECTIONS_FOR_LOCK = 3  # Majority vote after 3 detections
TARGET_SAMPLE_RATE = 16000


@dataclass
class RoutingState:
    start_time: float = field(default_factory=time.time)
    language_counts: Counter = field(default_factory=Counter)
    detection_count: int = 0
    locked_language: str | None = None
    use_aibharath: bool = False


class STT(stt.STT):
    """Routes STT based on detected or fixed language using VoxLingua107."""

    def __init__(
        self,
        *,
        whisper_url: str = "http://localhost:8003",
        aibharath_url: str = "http://localhost:8002",
        language_mode: str = "auto",  # "auto" or specific: "kn", "hi", "en", etc.
        translate_to_english: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._whisper = WhisperSTT(base_url=whisper_url, language="auto")
        self._aibharath = AIBharathSTT(
            base_url=aibharath_url,
            language="hi",
            translate_to_english=translate_to_english
        )
        self._aibharath_url = aibharath_url
        self._language_mode = language_mode
        self._state = RoutingState()
        self._session: aiohttp.ClientSession | None = None

        # If fixed language, lock immediately
        if language_mode != "auto":
            self._state.locked_language = language_mode
            self._state.use_aibharath = language_mode in INDIAN_LANGUAGES
            if self._state.use_aibharath:
                self._aibharath.update_options(language=language_mode)
            logger.info(f"Fixed language mode: {language_mode} (AIBharath: {self._state.use_aibharath})")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def aclose(self) -> None:
        await self._whisper.aclose()
        await self._aibharath.aclose()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _detect_language_voxlingua(self, buffer: AudioBuffer) -> str | None:
        """Detect language using VoxLingua107 via HTTP API."""
        frames = buffer if isinstance(buffer, list) else [buffer]
        audio_chunks = []
        sample_rate = TARGET_SAMPLE_RATE

        for frame in frames:
            if isinstance(frame, rtc.AudioFrame):
                sample_rate = frame.sample_rate
                pcm = np.frombuffer(bytes(frame.data), dtype=np.int16)
                if frame.num_channels == 2:
                    pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
                audio_chunks.append(pcm)

        if not audio_chunks:
            return None

        audio_data = np.concatenate(audio_chunks)

        # Resample to 16kHz if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = rtc.AudioResampler(
                input_rate=sample_rate,
                output_rate=TARGET_SAMPLE_RATE,
                num_channels=1
            )
            frame = rtc.AudioFrame(
                data=audio_data.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_data)
            )
            resampled = resampler.push(frame) + resampler.flush()
            if resampled:
                audio_data = np.frombuffer(b''.join(bytes(f.data) for f in resampled), dtype=np.int16)

        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(TARGET_SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        wav_buffer.seek(0)

        # Call VoxLingua107 API
        session = self._ensure_session()
        try:
            form = aiohttp.FormData()
            form.add_field('file', wav_buffer, filename='audio.wav', content_type='audio/wav')

            async with session.post(
                f"{self._aibharath_url}/detect-language",
                data=form,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    lang = data.get("language", "en")
                    confidence = data.get("confidence", 0.0)
                    is_indian = data.get("is_indian", False)
                    logger.info(f"VoxLingua107: {lang} (confidence: {confidence:.2%}, indian: {is_indian})")
                    return lang
                else:
                    logger.warning(f"VoxLingua107 API error: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"VoxLingua107 detection failed: {e}")
            return None

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        state = self._state

        # If language already locked, use appropriate STT
        if state.locked_language:
            return await self._transcribe_with_locked(buffer, conn_options)

        # Detection phase: Use VoxLingua107 for language detection
        detected = await self._detect_language_voxlingua(buffer)
        is_indian = detected in INDIAN_LANGUAGES if detected else False

        if detected:
            state.language_counts[detected] += 1
            state.detection_count += 1
            logger.info(f"Detection #{state.detection_count}: {detected} (indian: {is_indian}, counts: {dict(state.language_counts)})")

            # Check if we have enough detections for majority vote
            if state.detection_count >= MIN_DETECTIONS_FOR_LOCK:
                self._lock_language()

        # Route based on current detection (not just after locking)
        # This ensures correct STT is used even during detection phase
        if state.locked_language:
            return await self._transcribe_with_locked(buffer, conn_options)
        elif is_indian:
            # Indian language detected → use AIBharath Conformer
            self._aibharath.update_options(language=detected)
            logger.info(f"Routing to AIBharath (detected: {detected})")
            return await self._aibharath._recognize_impl(buffer, conn_options=conn_options)
        else:
            # Non-Indian or unknown → try Whisper, fallback to AIBharath
            try:
                return await self._whisper._recognize_impl(buffer, conn_options=conn_options)
            except Exception as e:
                logger.warning(f"Whisper failed, falling back to AIBharath: {e}")
                return await self._aibharath._recognize_impl(buffer, conn_options=conn_options)

    def _lock_language(self) -> None:
        """Lock to a language based on majority vote, preferring Indian languages on tie."""
        state = self._state
        counts = state.language_counts

        # Separate Indian and non-Indian languages
        indian_counts = {k: v for k, v in counts.items() if k in INDIAN_LANGUAGES}
        non_indian_counts = {k: v for k, v in counts.items() if k not in INDIAN_LANGUAGES}

        # Get max votes for each category
        max_indian = max(indian_counts.values()) if indian_counts else 0
        max_non_indian = max(non_indian_counts.values()) if non_indian_counts else 0

        # Prefer Indian language if it has equal or more votes
        if max_indian >= max_non_indian and indian_counts:
            most_common = max(indian_counts, key=indian_counts.get)
            state.use_aibharath = True
        elif non_indian_counts:
            most_common = max(non_indian_counts, key=non_indian_counts.get)
            state.use_aibharath = False
        else:
            most_common = counts.most_common(1)[0][0]
            state.use_aibharath = most_common in INDIAN_LANGUAGES

        state.locked_language = most_common

        if state.use_aibharath:
            self._aibharath.update_options(language=most_common)
            logger.info(f"Locked to AIBharath STT (language: {most_common}, votes: {dict(counts)})")
        else:
            logger.info(f"Locked to Whisper STT (language: {most_common}, votes: {dict(counts)})")

    async def _transcribe_with_locked(
        self,
        buffer: AudioBuffer,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if self._state.use_aibharath:
            return await self._aibharath._recognize_impl(buffer, conn_options=conn_options)
        return await self._whisper._recognize_impl(buffer, conn_options=conn_options)

    @property
    def detected_language(self) -> str | None:
        """Get the locked language after detection."""
        return self._state.locked_language

    @property
    def is_indian_language(self) -> bool:
        """Check if detected language is Indian."""
        return self._state.use_aibharath
