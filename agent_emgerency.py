import spacy
import datetime
from pathlib import Path
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, JobContext, WorkerOptions, cli, function_tool, room_io, ConversationItemAddedEvent
from livekit.plugins import silero, noise_cancellation, openai

# Custom plugin imports for Indian Language support
from plugins.tts.indic_tts import TTS as IndicTTS, SUPPORTED_LANGUAGES 
from plugins.stt.aibharath_conformer_stt import STT as IndicConformerSTT

nlp = spacy.load("en_core_web_sm")  # Load once globally


LOG_FILE = Path("conversation_log.txt")

# Load environment variables (e.g., LIVEKIT_URL, LIVEKIT_API_KEY)
load_dotenv()

DEFAULT_LANGUAGE = "en"

# Language name to code mapping
# Maps various inputs (English names, Native script, ISO codes) to standard 2-letter codes.
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
    An AI Agent acting as an Emergency Call Taker for India.
    
    This agent manages the conversation flow, adheres to specific rules regarding
    sentence length and emergency protocols, and handles dynamic language switching
    via function calling.
    """

    def __init__(self):
        # Dynamically generate the list of supported languages for the system prompt
        lang_list = ", ".join([info["name"] for info in SUPPORTED_LANGUAGES.values()])
        
        # Initialize the parent Agent with specific instructions (System Prompt)
        super().__init__( # STEP 1 - LANGUAGE:
                        # When caller says their language preference, call set_language tool ONCE.
                        # Supported: {lang_list}
            instructions=f"""You are an EMERGENCY CALL TAKER for India Emergency Helpline.
                        

                        STEP 2 - EMERGENCY INFO (in caller's language):
                        1. Ask what happened
                        2. Ask location/address
                        3. Ask caller name
                        4. Confirm and dispatch help

                        RULES:
                        - Call set_language tool ONLY ONCE at start
                        - ONE short sentence (max 10 words)
                        - ONE question at a time
                        - Respond in caller's selected language."""
        )
        self.current_language = DEFAULT_LANGUAGE
        self._language_set = False
    
    # @function_tool()
    # async def set_language(self, context: agents.RunContext, language: str):
    #     """
    #     Tool to change the conversation language.
        
    #     This updates the Text-To-Speech (TTS) and Speech-To-Text (STT) configurations
    #     in real-time during the active session.

    #     Args:
    #         context (agents.RunContext): The current agent execution context.
    #         language (str): The language requested by the user (e.g., "Hindi", "hi", "हिंदी").

    #     Returns:
    #         str: Confirmation message for the LLM.
    #     """
    #     # Normalize input to 2-letter language code
    #     lang_code = LANGUAGE_MAP.get(language.lower().strip(), "en")
    #     session = context.session

    #     # Update TTS settings if available
    #     if session.tts:
    #         session.tts.update_options(language=lang_code)
        
    #     # Update STT settings if available (disable translation to keep native script)
    #     if session.stt:
    #         session.stt.update_options(language=lang_code, translate_to_english=True)
        
    #     self.current_language = lang_code
    #     lang_name = SUPPORTED_LANGUAGES[lang_code]["name"]
        
    #     # Audio feedback to the user confirming the switch
    #     # await session.say(f"Switched to {lang_name}. What next?", allow_interruptions=False)
        
    #     return f"Now using {lang_name}."


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit Worker.
    
    Initializes the session components (STT, LLM, TTS, VAD) and starts the agent
    in the connected room.
    """
    
    # Configure the Agent Session components
    session = AgentSession(
        # Custom STT for Indian languages
        stt=IndicConformerSTT(language=DEFAULT_LANGUAGE, translate_to_english=True),
        
        llm = "openai/gpt-4.1-mini", 
        
        # Custom TTS for Indian languages
        tts=IndicTTS(language=DEFAULT_LANGUAGE, speaker="female"),
        
        # Voice Activity Detection settings
        vad=silero.VAD.load(),
        turn_detection="vad",
    )
    
    @session.on("user_input_transcribed")
    def on_transcript(transcript):
        if transcript.is_final:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("user_speech_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {transcript.transcript}\n")

    def log_turn(speaker: str, text: str):
        if not text.strip():
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract entities
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(f"{ent.text}({ent.label_})")
        entities_str = "; ".join(entities) if entities else "None"
        
        line = f"[{timestamp}] [{speaker}] {text} | ENTITIES: {entities_str}\n"
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line.strip())


    @session.on("user_input_transcribed")
    def on_user_transcript(evt):
        if evt.is_final:
            log_turn("USER", evt.transcript)

    @session.on("conversation_item_added")
    def on_conv_item(evt):
        if hasattr(evt, 'item') and evt.item.role == 'assistant' and evt.item.text_content:
            log_turn("AGENT", evt.item.text_content)

    
    assistant = IndicAssistant()
        
    # Start the session and connect to the room
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Apply background noise cancellation
                noise_cancellation=lambda p: noise_cancellation.BVC()  # Simplified
            ),
        ),
    )
    
    # Send the initial greeting to prompt the user for language selection
    await session.say("Hello. Language? English/Hindi/Kannada/Malayalam/Marathi/Tamil/Telugu.", allow_interruptions=True)
    await ctx.connect()
    
    
if __name__ == "__main__":
    # Run the worker application
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    
    
    
# Local LLM configuration (using Ollama)
# llm=openai.LLM(
#     base_url="http://192.168.1.120:11434/v1",
#     api_key="unused", # Not required for Ollama
#     model="llama3.1:8b",
#     extra_body={"keep_alive": "24h"} # Keep model loaded in memory
# ),