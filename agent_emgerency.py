from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import spacy
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli, function_tool, room_io
from livekit.plugins import silero, noise_cancellation, openai

from db_mssql import create_incident, write_call_metrics
from prompts import SYSTEM_PROMPT
from schemas import CallInfo, CallState, ExtractionConfidence

from plugins.tts.indic_tts import TTS as IndicTTS, SUPPORTED_LANGUAGES

DEFAULT_LANGUAGE = "en"
LOG_FILE = Path("conversation_log.txt")
STATE_LOG = Path("extraction_state.jsonl")
FULL_TRANSCRIPT_LOG = Path("full_transcript.txt")

load_dotenv()
nlp = spacy.load("en_core_web_sm")  # global load

LANGUAGE_MAP = {
    "english": "en", "en": "en",
    "hindi": "hi", "hi": "hi", "हिंदी": "hi",
    "kannada": "kn", "kn": "kn", "ಕನ್ನಡ": "kn",
    "malayalam": "ml", "ml": "ml", "മലയാളം": "ml",
    "marathi": "mr", "mr": "mr", "मराठी": "mr",
    "tamil": "ta", "ta": "ta", "தமிழ்": "ta",
    "telugu": "te", "te": "te", "తెలుగు": "te",
}


def ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_text(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_turn(speaker: str, text: str) -> None:
    if not text or not text.strip():
        return
    doc = nlp(text)
    ents = [f"{e.text}({e.label_})" for e in doc.ents]
    entities_str = "; ".join(ents) if ents else "None"
    line = f"[{ts()}] [{speaker}] {text} | ENTITIES: {entities_str}"
    append_text(LOG_FILE, line)


def get_state(session: AgentSession) -> CallState:
    st = session.userdata.get("state")
    if isinstance(st, CallState):
        return st
    # create new state
    st = CallState()
    session.userdata["state"] = st
    return st


class EmergencyCallAgent(agents.Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)
        self._language_set = False

    @function_tool()
    async def set_language(self, context: agents.RunContext, language: str) -> str:
        st = get_state(context.session)
        if self._language_set:
            return "Language already set."

        lang_code = LANGUAGE_MAP.get(language.lower().strip(), DEFAULT_LANGUAGE)
        st.language = lang_code
        self._language_set = True

        # Update TTS/STT if your plugins support update_options
        if context.session.tts:
            try:
                context.session.tts.update_options(language=lang_code)
            except Exception:
                pass

        if context.session.stt:
            try:
                context.session.stt.update_options(language=lang_code)
            except Exception:
                pass

        return f"Language set to {lang_code}."

    @function_tool()
    async def ner_hint(self, context: agents.RunContext, text: str) -> str:
        """
        Lightweight NER hint (English) to support extraction.
        """
        doc = nlp(text)
        out = [{"text": e.text, "label": e.label_} for e in doc.ents]
        return json.dumps(out, ensure_ascii=False)

    @function_tool()
    async def extract_update(self, context: agents.RunContext, payload_json: str) -> str:
        """
        LLM calls this once per user turn with best-effort JSON for:
        call_type, intent, incident_type, location, caller_name, confidence {overall, fields}.
        """
        st = get_state(context.session)
        st.metrics.extraction_updates += 1

        try:
            payload = json.loads(payload_json)
        except Exception:
            payload = {}

        # Merge into CallInfo (best-effort)
        old = st.info.model_dump()
        merged = {**old, **payload}

        # Fix confidence nested object
        conf = merged.get("confidence") or {}
        if isinstance(conf, dict):
            merged["confidence"] = ExtractionConfidence(
                overall=float(conf.get("overall", st.info.confidence.overall or 0.0) or 0.0),
                fields=dict(conf.get("fields", st.info.confidence.fields or {}) or {}),
            ).model_dump()

        st.info = CallInfo.model_validate(merged)

        # Persist state snapshot (for demo + accuracy scoring later)
        append_text(STATE_LOG, json.dumps({
            "t": ts(),
            "call_id": st.call_id,
            "stage": st.stage,
            "language": st.language,
            "info": st.info.model_dump(),
        }, ensure_ascii=False))

        # Auto-stage transitions
        if st.stage == "need_language":
            st.stage = "collecting"
        if st.info.confirmed and st.stage != "created":
            st.stage = "confirming"

        return "OK"

    @function_tool()
    async def mark_confirmed(self, context: agents.RunContext, confirmed: bool) -> str:
        st = get_state(context.session)
        st.info.confirmed = bool(confirmed)
        st.stage = "confirming" if confirmed else "collecting"
        return "OK"

    @function_tool()
    async def create_incident(self, context: agents.RunContext) -> str:
        """
        Create incident in MSSQL. Uses pyodbc commit pattern. [web:35]
        """
        st = get_state(context.session)
        conn_str = (context.session.userdata.get("MSSQL_CONN_STR") or "").strip()
        if not conn_str:
            return "Missing MSSQL_CONN_STR."

        # Pull transcript text you logged
        transcript_text = context.session.userdata.get("full_transcript", "")

        incident_id = create_incident(
            conn_str,
            call_id=st.call_id,
            language=st.language,
            info=st.info,
            transcript_text=transcript_text,
        )

        st.metrics.incident_created = True
        st.metrics.incident_id = incident_id
        st.stage = "created"
        return f"INCIDENT_CREATED:{incident_id}"

    @function_tool()
    async def disconnect(self, context: agents.RunContext) -> str:
        """
        Graceful termination: session emits a `close` event on shutdown. [web:12]
        """
        st = get_state(context.session)
        st.stage = "ended"
        try:
            maybe_coro = context.session.close()
            if hasattr(maybe_coro, "__await__"):
                await maybe_coro
        except Exception:
            pass
        return "DISCONNECTED"


async def entrypoint(ctx: JobContext):
    # ---- STT: Whisper-style with VAD buffering for non-streaming STT. [web:26]
    whisper = openai.STT()  # uses OPENAI_API_KEY/base_url if configured
    vad_for_stt = silero.VAD.load(min_speech_duration=0.1, min_silence_duration=0.5)
    stt = agents.stt.StreamAdapter(whisper, vad_for_stt.stream())  # Whisper + VAD chunking [web:26]

    # ---- Session
    session = AgentSession(
        stt=stt,
        llm=openai.LLM(
            base_url="http://192.168.1.120:11434/v1",
            api_key="unused",
            model="llama3.1:8b",
            extra_body={"keep_alive": "24h"},
        ),
        # Swap IndicTTS -> IndicParlerTTS when your plugin is ready.
        tts=IndicTTS(language=DEFAULT_LANGUAGE, speaker="female"),
        vad=silero.VAD.load(),
        turn_detection="vad",
    )

    # Attach initial state + secrets
    st = CallState()
    session.userdata["state"] = st
    # put MSSQL conn str into userdata so tools can access it
    # Example: Driver={ODBC Driver 18 for SQL Server};Server=...;Database=...;UID=...;PWD=...
    session.userdata["MSSQL_CONN_STR"] = (ctx.env.get("MSSQL_CONN_STR") or "").strip()
    session.userdata["full_transcript"] = ""

    # ---- Observability hooks (events are part of AgentSession) [web:12]
    @session.on("user_input_transcribed")
    def on_user_transcript(evt):
        # evt has interim/final transcripts depending on provider
        # You used evt.is_final earlier; keep that convention:
        if getattr(evt, "is_final", False):
            st.metrics.stt_final_count += 1
            st.metrics.user_turns += 1
            log_turn("USER", evt.transcript)
            append_text(FULL_TRANSCRIPT_LOG, f"[{ts()}] USER: {evt.transcript}")
            session.userdata["full_transcript"] += f"\nUSER: {evt.transcript}"
        else:
            st.metrics.stt_interim_count += 1

    @session.on("conversation_item_added")
    def on_item(evt):
        try:
            item = evt.item
            if item.role == "assistant" and getattr(item, "text_content", None):
                st.metrics.agent_turns += 1
                log_turn("AGENT", item.text_content)
                append_text(FULL_TRANSCRIPT_LOG, f"[{ts()}] AGENT: {item.text_content}")
                session.userdata["full_transcript"] += f"\nAGENT: {item.text_content}"
        except Exception:
            pass

    @session.on("close")
    def on_close(evt):
        st.metrics.ended_at_unix = time.time()
        conn_str = (session.userdata.get("MSSQL_CONN_STR") or "").strip()
        if conn_str:
            try:
                write_call_metrics(conn_str, st.metrics)
            except Exception:
                pass

    # Start session (AgentSession orchestrates pipeline + emits events) [web:12]
    agent = EmergencyCallAgent()
    await session.start(
        room=ctx.room,
        agent=agent,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda _: noise_cancellation.BVC(),
            ),
        ),
    )

    # Prompt for language at beginning
    await session.say(
        "Hello. Language?",
        allow_interruptions=True,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))