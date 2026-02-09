import spacy
import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli, function_tool, room_io
from livekit.plugins import silero, noise_cancellation, openai

# Database Library for MSSQL
try:
    import pymssql
    MSSQL_AVAILABLE = True
except ImportError:
    MSSQL_AVAILABLE = False
    print("Warning: pymssql not installed. MSSQL features will be simulated.")

# Custom plugin imports
from plugins.tts.indic_tts import TTS as IndicTTS, SUPPORTED_LANGUAGES 
from plugins.stt.aibharath_conformer_stt import STT as IndicConformerSTT

# Load Spacy
nlp = spacy.load("en_core_web_sm")

# Configuration
LOG_FILE = Path("conversation_log.txt")
EVAL_FILE = Path("evaluation_metrics.jsonl")
load_dotenv()

DEFAULT_LANGUAGE = "en"

# Database Configuration (Load from .env or use defaults)
DB_CONFIG = {
    'server': os.getenv('MSSQL_SERVER', 'localhost'),
    'user': os.getenv('MSSQL_USER', 'sa'),
    'password': os.getenv('MSSQL_PASSWORD', 'password'),
    'database': os.getenv('MSSQL_DATABASE', 'EmergencyDB')
}

LANGUAGE_MAP = {
    "english": "en", "en": "en",
    "hindi": "hi", "hi": "hi", "हिंदी": "hi",
    "kannada": "kn", "kn": "kn", "ಕನ್ನಡ": "kn",
    "malayalam": "ml", "ml": "ml", "മലയാളം": "ml",
    "marathi": "mr", "mr": "mr", "मराठी": "mr",
    "tamil": "ta", "ta": "ta", "தமிழ்": "ta",
    "telugu": "te", "te": "te", "తెలుగు": "te",
}

class DatabaseManager:
    """Handles MSSQL Operations"""
    @staticmethod
    def insert_incident(data: dict):
        if not MSSQL_AVAILABLE:
            return "Simulated DB Write (pymssql not installed)"

        try:
            conn = pymssql.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            query = """
                INSERT INTO Incidents 
                (TicketID, Timestamp, CallerName, Location, IncidentType, Classification, Confidence, Description, Language)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                data['ticket_id'],
                data['timestamp'],
                data['name'],
                data['location'],
                data['type'],
                data['classification'],
                data['confidence'],
                data['description'],
                data['language']
            ))
            conn.commit()
            conn.close()
            return "Success: Written to MSSQL"
        except Exception as e:
            return f"DB Error: {str(e)}"

class IndicAssistant(agents.Agent):
    def __init__(self):
        lang_list = ", ".join([info["name"] for info in SUPPORTED_LANGUAGES.values()])
        
        super().__init__(
            instructions=f"""You are an EMERGENCY CALL TAKER for the India Emergency Helpline.

            STRICT WORKFLOW SEQUENCE:
            1. INTENT CLASSIFICATION:
               - Listen to the user.
               - Call 'detect_call_intent' to determine the nature of the call and your confidence.
               - If confidence is low (<0.7), ask clarifying questions.
               - If it's an Inquiry, answer briefly and ask if they have an emergency.
               - If Emergency, proceed to Step 2.

            2. LANGUAGE SETUP:
               - Call 'set_language' ONCE.

            3. MANDATORY EXTRACTION:
               - Collect: Name, Location, Incident Type.
               - Ask ONE short question at a time (max 10 words).

            4. SUBMISSION & ACKNOWLEDGMENT:
               - Call 'submit_emergency_report' with all details.

            5. DISCONNECT:
               - After submission success, read the Ticket ID.
               - Call 'disconnect_call' to end the session.

            RULES:
            - Be calm and empathetic.
            - High urgency requires faster, shorter sentences.
            - Do not hang up until the report is submitted.
            """
        )
        self.current_language = DEFAULT_LANGUAGE
        self._language_set = False
        self.call_confidence = 0.0
        self.call_type = "UNKNOWN"

    @function_tool()
    async def detect_call_intent(self, context: agents.RunContext, user_statement: str):
        """
        Analyzes the user's initial statement to classify the call type and confidence.
        """
        # In a real scenario, this might use a separate classifier model. 
        # Here we use the LLM's internal reasoning to populate the tool output.
        
        # Simulated Logic for the demo: 
        # We rely on the LLM to fill these arguments correctly based on its internal understanding
        # of the conversation history.
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
        confidence_score: float, # Added for evaluation
        description: str
    ):
        """
        Submits the final report to the database and triggers the disconnect sequence.
        
        Args:
            caller_name: Name of the caller.
            location: Incident address.
            incident_type: Fire, Medical, etc.
            call_classification: EMERGENCY, INQUIRY, or REPORT.
            confidence_score: 0.0 to 1.0 confidence in the classification.
            description: Summary of events.
        """
        ticket_id = f"INC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        db_data = {
            "ticket_id": ticket_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "name": caller_name,
            "location": location,
            "type": incident_type,
            "classification": call_classification,
            "confidence": confidence_score,
            "description": description,
            "language": self.current_language
        }

        # Write to DB
        db_result = DatabaseManager.insert_incident(db_data)
        
        # Log Evaluation Metrics
        with EVAL_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(db_data, ensure_ascii=False) + "\n")

        if "Success" in db_result or "Simulated" in db_result:
            return f"Report {ticket_id} submitted successfully. Help dispatched."
        else:
            return f"System error: {db_result}"

    @function_tool()
    async def disconnect_call(self, context: agents.RunContext):
        """
        Gracefully disconnects the call after the report is submitted.
        """
        # Log the end of the call
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] SYSTEM: Call disconnected by Agent.\n")
        
        # Actual disconnect logic
        # Note: In some LiveKit versions, we might just return a message and let the user hang up,
        # or we can actively close the room if permissions allow.
        try:
            # This sends a signal to the worker to close the connection
            await context.room.disconnect()
        except Exception:
            pass # Room might already be closing
            
        return "Disconnecting call. Stay safe."

async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=IndicConformerSTT(language=DEFAULT_LANGUAGE, translate_to_english=True),
        llm="openai/gpt-4o-mini", 
        tts=IndicTTS(language=DEFAULT_LANGUAGE, speaker="female"),
        vad=silero.VAD.load(),
        turn_detection="vad",
    )
    
    # --- Advanced Logging & Evaluation ---

    def log_event(event_type: str, data: dict):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {event_type} | {json.dumps(data)}\n"
        
        # Human readable log
        with LOG_FILE.open("a", encoding="utf-8") as f:
            if "text" in data:
                # NER Analysis on text
                doc = nlp(data["text"])
                entities = [(e.text, e.label_) for e in doc.ents]
                f.write(f"[{timestamp}] [{event_type}] {data['text']} | ENTITIES: {entities}\n")
            else:
                f.write(log_entry)

    @session.on("user_input_transcribed")
    def on_user(evt):
        if evt.is_final:
            log_event("USER", {"text": evt.transcript})

    @session.on("conversation_item_added")
    def on_agent(evt):
        if hasattr(evt, 'item'):
            if evt.item.type == "function_call":
                # Log tool calls for accuracy evaluation
                args = evt.item.arguments
                log_event("TOOL_CALL", {"name": evt.item.name, "args": args})
            
            elif evt.item.role == 'assistant' and evt.item.text_content:
                log_event("AGENT", {"text": evt.item.text_content})

    # --- Start Agent ---
    
    assistant = IndicAssistant()
        
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
    
    # Initial prompt to trigger the workflow
    await session.say("Emergency Helpline. What is your emergency?", allow_interruptions=True)
    
    
if __name__ == "__main__":
    # Initialize DB Table Script (Optional, for setup)
    # print("Ensure MSSQL table 'Incidents' exists with columns matching the insert query.")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))