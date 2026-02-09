import spacy
import datetime
from pathlib import Path
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli, function_tool, room_io
from livekit.plugins import silero, noise_cancellation, openai

# Custom plugin imports for Indian Language support
from plugins.tts.indic_tts import TTS as IndicTTS, SUPPORTED_LANGUAGES 
from plugins.stt.aibharath_conformer_stt import STT as IndicConformerSTT

# Load Spacy NER model for entity extraction (Locations, Names, etc.)
nlp = spacy.load("en_core_web_sm")

LOG_FILE = Path("conversation_log.txt")
load_dotenv()

DEFAULT_LANGUAGE = "en"

# Language name to code mapping
LANGUAGE_MAP = {
    "english": "en", "en": "en",
    "hindi": "hi", "hi": "hi", "हिंदी": "hi",
    "kannada": "kn", "kn": "kn", "ಕನ್ನಡ": "kn",
    "malayalam": "ml", "ml": "ml", "മലയാളം": "ml",
    "marathi": "mr", "mr": "mr", "मराठी": "mr",
    "tamil": "ta", "ta": "ta", "தமிழ்": "ta",
    "telugu": "te", "te": "te", "తెలుగు": "te",
}

class IndicAssistant(agents.Agent):
    """
    Emergency Call Taker for India.
    Features: Dynamic Language Switching, Entity Extraction, Incident Classification.
    """

    def __init__(self):
        # Generate list of languages for the prompt
        lang_list = ", ".join([info["name"] for info in SUPPORTED_LANGUAGES.values()])
        
        super().__init__(
            instructions=f"""You are an EMERGENCY CALL TAKER for the India Emergency Helpline.

            WORKFLOW:
            1. LANGUAGE DETECTION: 
               - Listen to the user's first sentence.
               - Call the 'set_language' tool immediately with the detected language.
               - ONLY call this ONCE.
               - Supported languages: {lang_list}.

            2. INFORMATION GATHERING (Mandatory Extraction):
               - Ask short, clear questions (max 10 words).
               - You MUST collect: 
                 a) Name of caller
                 b) Location/Address (Be specific, ask for landmarks if needed)
                 c) Incident Type (Fire, Accident, Medical, etc.)
               - Do not move to step 3 until you have this info.

            3. CLASSIFICATION & SUBMISSION:
               - Determine Call Classification: "EMERGENCY", "INQUIRY", or "REPORT".
               - Call the 'submit_emergency_report' tool with ALL gathered details.

            4. CLOSING:
               - After the tool returns success, read the confirmation ID to the user.
               - Say a polite goodbye and wait for the user to hang up.

            RULES:
            - Stay calm and professional.
            - If the user is panicked, speak slowly and clearly.
            - If user provides multiple sentences, break down your responses.
            - NEVER ask more than ONE question at a time.
            """
        )
        self.current_language = DEFAULT_LANGUAGE
        self._language_set = False

    @function_tool()
    async def set_language(self, context: agents.RunContext, language: str):
        """
        Sets the language for the session (STT and TTS). 
        MUST be called only once at the start.
        """
        if self._language_set:
            return "Language is already set. Proceed with the emergency details."

        # Normalize input
        lang_code = LANGUAGE_MAP.get(language.lower().strip(), "en")
        session = context.session

        # Update TTS
        if session.tts:
            try:
                # Update plugin options if supported
                session.tts.update_options(language=lang_code)
            except AttributeError:
                print(f"TTS plugin does not support dynamic update, attempting to use default behavior for {lang_code}")

        # Update STT
        if session.stt:
            try:
                session.stt.update_options(language=lang_code, translate_to_english=True)
            except AttributeError:
                 print(f"STT plugin does not support dynamic update, using default behavior for {lang_code}")

        self.current_language = lang_code
        self._language_set = True
        lang_name = SUPPORTED_LANGUAGES.get(lang_code, {}).get("name", language)
        
        return f"Language switched to {lang_name}. Proceeding to collect emergency details."

    @function_tool()
    async def submit_emergency_report(
        self, 
        context: agents.RunContext, 
        caller_name: str, 
        location: str, 
        incident_type: str, 
        call_classification: str,
        description: str
    ):
        """
        Call this tool after extracting all mandatory information.
        This classifies the call and logs the incident to the database.
        
        Args:
            caller_name: Name of the person calling.
            location: Specific address or location of the incident.
            incident_type: E.g., 'Fire', 'Car Accident', 'Heart Attack'.
            call_classification: Must be 'EMERGENCY', 'INQUIRY', or 'REPORT'.
            description: A brief summary of what happened.
        """
        
        # 1. Validation
        valid_classifications = ["EMERGENCY", "INQUIRY", "REPORT"]
        if call_classification.upper() not in valid_classifications:
            return f"Error: Invalid classification. Must be one of {valid_classifications}."

        # 2. Simulate MSSQL Write / Database Insertion
        # In production, replace this with actual pyodbc/pymssql connection logic
        ticket_id = f"INC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        db_record = {
            "ticket_id": ticket_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "name": caller_name,
            "location": location,
            "type": incident_type,
            "classification": call_classification.upper(),
            "description": description,
            "language": self.current_language
        }

        # Simulating DB latency
        import asyncio
        await asyncio.sleep(0.5)
        
        # Log to file for auditing (Extraction Accuracy Evaluation)
        with open("incident_db_dump.jsonl", "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(db_record, ensure_ascii=False) + "\n")

        return f"Success. Incident {ticket_id} recorded as {call_classification.upper()}. Help is being dispatched to {location}."


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit Worker.
    """
    
    session = AgentSession(
        stt=IndicConformerSTT(language=DEFAULT_LANGUAGE, translate_to_english=True),
        llm="openai/gpt-4o-mini", # Updated to a faster/cost-effective model
        tts=IndicTTS(language=DEFAULT_LANGUAGE, speaker="female"),
        vad=silero.VAD.load(),
        turn_detection="vad",
    )
    
    # --- Logging Logic ---

    def log_to_file(speaker: str, text: str, metadata: dict = None):
        if not text.strip():
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # NER Extraction (Spacy)
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(f"{ent.text}({ent.label_})")
        
        entities_str = "; ".join(entities) if entities else "None"
        
        # Prepare log line
        meta_str = f" | META: {metadata}" if metadata else ""
        line = f"[{timestamp}] [{speaker}] {text} | NER: {entities_str}{meta_str}\n"
        
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line.strip())

    @session.on("user_input_transcribed")
    def on_user_transcript(evt):
        if evt.is_final:
            log_to_file("USER", evt.transcript)

    @session.on("conversation_item_added")
    def on_conv_item(evt):
        # Log Agent responses
        if hasattr(evt, 'item') and evt.item.role == 'assistant' and evt.item.text_content:
            log_to_file("AGENT", evt.item.text_content)
        
        # Evaluation: Track Tool Calls for accuracy metrics
        if hasattr(evt, 'item') and evt.item.type == "function_call":
            tool_name = evt.item.name
            args = evt.item.arguments
            log_to_file("SYSTEM", f"Tool Called: {tool_name}", metadata=args)

    # --- Start Session ---

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
    
    # Initial Greeting
    # We let the agent handle the flow, but we can nudge it if needed via say.
    # However, the system prompt is usually enough to trigger the first interaction 
    # if the VAD detects the user. 
    # To be safe and ensure the Language step happens:
    await session.say("Hello. Please tell me your preferred language.", allow_interruptions=True)
    
    
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))