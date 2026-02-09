import spacy
import datetime
import json
import os
import sqlite3
from pathlib import Path
from prompt import instructions
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli, function_tool, room_io
from livekit.plugins import silero, noise_cancellation

# Custom plugin imports
from plugins.tts.indic_tts import TTS as IndicTTS, SUPPORTED_LANGUAGES 
from plugins.stt.aibharath_conformer_stt import STT as IndicConformerSTT

# Load Spacy
nlp = spacy.load("en_core_web_sm")

# Configuration
LOG_DIR = Path("conversation_logs")
JSON_DIR = Path("conversation_json")
DB_FILE = Path("emergency_data.db")

# Create directories if they don't exist
LOG_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

load_dotenv()

DEFAULT_LANGUAGE = "en"

LANGUAGE_MAP = {
    "english": "en", "en": "en",
    "hindi": "hi", "hi": "hi", "हिंदी": "hi",
    "kannada": "kn", "kn": "kn", "ಕನ್ನಡ": "kn",
    "malayalam": "ml", "ml": "ml", "മലയാളം": "ml",
    "marathi": "mr", "mr": "mr", "मराठी": "mr",
    "tamil": "ta", "ta": "ta", "தமிழ்": "ta",
    "telugu": "te", "te": "te", "తెలుగు": "te",
}

class SQLiteManager:
    """Handles SQLite Operations"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE,
                timestamp TEXT,
                caller_name TEXT,
                location TEXT,
                incident_type TEXT,
                classification TEXT,
                confidence REAL,
                priority TEXT,
                sentiment TEXT,
                description TEXT,
                language TEXT,
                json_file_path TEXT
            )
        """)
        conn.commit()
        conn.close()

    def insert_incident(self, data: dict, json_path: str):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                INSERT INTO incidents 
                (TicketID, Timestamp, CallerName, Location, IncidentType, Classification, Confidence, Priority, Sentiment, Description, Language, JsonFilePath)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                data['ticket_id'],
                data['timestamp'],
                data['name'],
                data['location'],
                data['type'],
                data['classification'],
                data['confidence'],
                data['priority'],
                data['sentiment'],
                data['description'],
                data['language'],
                json_path
            ))
            conn.commit()
            conn.close()
            return "Success: Written to SQLite"
        except Exception as e:
            return f"DB Error: {str(e)}"

# Initialize DB Manager globally
db_manager = SQLiteManager(DB_FILE)

class IndicAssistant(agents.Agent):
    def __init__(self):
        lang_list = ", ".join([info["name"] for info in SUPPORTED_LANGUAGES.values()])
        
        super().__init__(
            instructions=instructions
        )
        self.current_language = DEFAULT_LANGUAGE
        self._language_set = False
        self._report_submitted = False
        
        # Store conversation history here: [{"role": "user", "content": "..."}, ...]
        self.conversation_history = []

    @function_tool()
    async def detect_call_intent(self, context: agents.RunContext, user_statement: str):
        """
        Analyzes the user's initial statement to classify the call type and confidence.
        """
        return "Intent analyzed. Proceed based on the provided classification and confidence."

    @function_tool()
    async def set_language(self, context: agents.RunContext, language: str):
        if self._language_set:
            return "Language already set."
        
        lang_code = LANGUAGE_MAP.get(language.lower().strip(), "en")
        session = context.session

        if session.tts:
            try: session.tts.update_options(language=lang_code)
            except: pass
        
        if session.stt:
            try: session.stt.update_options(language=lang_code, translate_to_english=True)
            except: pass

        self.current_language = lang_code
        self._language_set = True
        return f"Language set to {language}. Proceeding."

    @function_tool()
    async def submit_emergency_report(
        self, 
        context: agents.RunContext, 
        caller_name: str, 
        location: str, 
        incident_type: str, 
        call_classification: str,
        confidence_score: float, 
        priority: str,        # e.g., High, Medium, Low
        sentiment: str,       # e.g., Panicked, Calm, Aggressive
        description: str
    ):
        """
        Submits the final report to SQLite and saves the full conversation JSON.
        THIS SHOULD BE CALLED ONLY ONCE.
        """
        if self._report_submitted:
            return "Report already submitted. Please disconnect."

        ticket_id = f"INC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        timestamp = datetime.datetime.now().isoformat()
        
        # Prepare Data for DB
        db_data = {
            "ticket_id": ticket_id,
            "timestamp": timestamp,
            "name": caller_name,
            "location": location,
            "type": incident_type,
            "classification": call_classification,
            "confidence": confidence_score,
            "priority": priority,
            "sentiment": sentiment,
            "description": description,
            "language": self.current_language
        }

        # 1. Save Conversation to JSON File
        json_filename = JSON_DIR / f"incident_{ticket_id}.json"
        conversation_data = {
            "metadata": db_data,
            "conversation_log": self.conversation_history
        }
        
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=4)

        # 2. Write to SQLite
        db_result = db_manager.insert_incident(db_data, str(json_filename))
        
        self._report_submitted = True

        if "Success" in db_result:
            return f"Report {ticket_id} submitted successfully. Help is on the way to {location}."
        else:
            return f"System error: {db_result}"

    @function_tool()
    async def disconnect_call(self, context: agents.RunContext):
        """
        Gracefully disconnects the call.
        """
        try:
            await context.room.disconnect()
        except Exception:
            pass
            
        return "Disconnecting call. Stay safe."

async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=IndicConformerSTT(language=DEFAULT_LANGUAGE, translate_to_english=True),
        llm="openai/gpt-4o-mini", 
        tts=IndicTTS(language=DEFAULT_LANGUAGE, speaker="female"),
        vad=silero.VAD.load(),
        turn_detection="vad",
    )
    
    assistant = IndicAssistant()
        
    # --- Logging Logic attached to Session ---

    @session.on("user_input_transcribed")
    def on_user(evt):
        if evt.is_final:
            # Append to conversation history
            assistant.conversation_history.append({
                "role": "user",
                "content": evt.transcript,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Optional: Keep a flat log file for quick debugging
            with open(LOG_DIR / "flat_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[USER] {evt.transcript}\n")

    @session.on("conversation_item_added")
    def on_agent(evt):
        if hasattr(evt, 'item'):
            # Log Agent Text
            if evt.item.role == 'assistant' and evt.item.text_content:
                assistant.conversation_history.append({
                    "role": "assistant",
                    "content": evt.item.text_content,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                with open(LOG_DIR / "flat_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[AGENT] {evt.item.text_content}\n")

            # Log Tool Calls (for debugging, not part of conversation json usually)
            if evt.item.type == "function_call":
                with open(LOG_DIR / "flat_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[TOOL] {evt.item.name}({evt.item.arguments})\n")

    # --- Start Agent ---
        
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda p: noise_cancellation.BVC()
            ),
        ),
    )
    
    await ctx.connect()
    
    # Initial prompt
    await session.say("Emergency Helpline. What is your emergency?", allow_interruptions=True)
    
    
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))