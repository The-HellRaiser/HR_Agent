import sounddevice as sd
import numpy as np
import wave
import subprocess
import time
import pyttsx3
import requests
import os
from zoomus import ZoomClient # Ensure you have installed python-zoomus or similar
import webbrowser
from urllib.parse import urlencode, urlparse, parse_qs
import collections # For deque
import re # For regex operations
# --- LangChain Imports ---
from langchain_ollama import ChatOllama  # Updated import
from langchain.agents import AgentExecutor, create_react_agent # Using ReAct agent
from langchain.tools import tool # Use LangChain's tool decorator
from langchain import hub # To pull standard prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # For custom prompts if needed
# --- End LangChain Imports ---

# litellm._turn_on_debug() # Keep if you were using it for other debugging

# --- Constants and Setup ---
# Zoom API credentials (still potentially useful for future enhancements)

JOB_DESCRIPTION = """
Senior Software Engineer Position:
- Lead development of AI-powered applications
- 5+ years of Python development experience
- Expert knowledge of machine learning frameworks (PyTorch, TensorFlow)
- Experience with cloud platforms (AWS/Azure/GCP)
- Strong system design and architecture skills
- Team leadership experience
"""

# Audio settings (Keep as is)
SAMPLERATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCKSIZE = 1024

# VAD settings (Keep as is)(modify if want to change the inteview length)
SILENCE_THRESHOLD = 50
SILENCE_DURATION_SEC = 2.0
MAX_RECORD_SEC = 15

# Whisper settings (Keep as is)
WHISPER_CPP_EXECUTABLE = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL_PATH = "/Users/researchassistant/HR Agent/HR_Agent/whisper.cpp/models/ggml-base.en.bin"

# --- Ollama Settings (for LangChain) ---
OLLAMA_BASE_URL = "http://localhost:11434" # Base URL for Ollama
OLLAMA_MODEL = "llama3.1:8b-instruct-q5_K_M" # Your Ollama model

# --- Global Variables (Keep as is) ---
audio_buffer = collections.deque()
recording_active = False
silent_frames = 0
frames_per_block = BLOCKSIZE
samples_per_second = SAMPLERATE
blocks_per_second = samples_per_second / frames_per_block
silence_blocks_needed = int(SILENCE_DURATION_SEC * blocks_per_second)

# --- Initialize LangChain LLM ---
# We'll pass this llm object to tools that need it
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2
)
# --- End LLM Initialization ---

# Add after other constants
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
# Ensure recordings directory exists
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# --- Tool Definitions (Using LangChain's @tool decorator) ---
# NOTE: Tool function logic remains IDENTICAL

@tool
def join_zoom_meeting() -> bool:
    """
    Joins a specific Zoom meeting by opening its URL.
    Opens the meeting URL in the default browser and waits briefly.

    Returns:
        bool: True if the browser was opened successfully, False otherwise.
    """
    try:
        # SECURITY NOTE: Hardcoding credentials/links is not recommended for production.
        # Consider environment variables or a config file.
        join_url = f"https://us04web.zoom.us/j/5234862235?pwd=MnA4akU5S1lQMHZLZlI4eU5IbFUrQT09"
        webbrowser.open(join_url)
        print(f"Attempting to join Zoom meeting via browser: {join_url}")
        time.sleep(5)  # Give browser/Zoom time to initiate
        return True
    except Exception as e:
        print(f"Error opening Zoom meeting URL: {e}")
        return False

@tool
def speak_reply(text: str) -> bool:
    """
    Speaks the given text using the pyttsx3 text-to-speech engine.
    Initializes the engine, sets properties (rate, potentially device), speaks, and cleans up.

    Args:
        text: The string of text to be spoken.

    Returns:
        bool: True if speech synthesis was successful, False if an error occurred.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)

        # --- Device Selection (Keep your specific logic here) ---
        # This part is environment-specific. Ensure the device index/name is correct.
        try:
            # Example: Manually set device - *ADJUST THIS INDEX/NAME*
            # Use `sd.query_devices()` to find the correct output device if needed.
            # This should be the VIRTUAL CABLE INPUT that Zoom listens to.
            # engine.setProperty('voice', 'com.apple.speech.synthesis.voice.daniel') # Example macOS voice
            # Setting specific device properties in pyttsx3 can be tricky and driver-dependent.
            # Often, setting the system's default output to the virtual cable is more reliable.
            print("pyttsx3 will use the system default audio output device.")
            # If you found a reliable way to set output device via pyttsx3, keep it here.
            devices = sd.query_devices() # Keep for debugging if needed
            print(devices)
            engine.setProperty('MacBook Pro Microphone', 1) # This line seems incorrect for *output* - REMOVED

        except Exception as dev_e:
            print(f"Warning: Could not query/set audio devices for pyttsx3: {dev_e}")
            print("Using system default output device.")
        # --- End Device Selection ---

        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine # Explicitly delete engine object
        time.sleep(0.5) # Small pause
        return True
    except Exception as e:
        print(f"Error in pyttsx3: {e}")
        return False



@tool
def generate_interview_question(previous_response: str = "None", question_history: list[str] = None) -> str:
    """
    Generates the next contextual interview question using the LLM based on the job description,
    the candidate's previous response, and the history of questions already asked.

    Args:
        previous_response (str): The candidate's last response. Defaults to 'None'.
        question_history (list[str]): List of questions already asked by the interviewer. Defaults to None.

    Returns:
        str: The generated interview question, or a fallback question if generation fails.
    """
    if question_history is None:
        question_history = []

    history_str = "\n".join([f"- {q}" for q in question_history]) if question_history else 'None'

    prompt_text = f"""
You are an AI interviewer conducting a screening for a Senior Software Engineer position.
Your goal is to ask relevant technical questions based on the job description and the conversation flow.

Job Description:
{JOB_DESCRIPTION}

Interview History (Questions Already Asked):
{history_str}

Candidate's Previous Response:
{previous_response}

Based on the job description, the interview history, and the candidate's *last* response, generate the *single next* relevant, natural, and conversational interview question.
- Focus on skills: Python, ML frameworks (PyTorch, TensorFlow), Cloud (AWS/Azure/GCP), System Design, Leadership.
- If the candidate gave a previous response, try to build upon it or ask about a *different* required skill area.
- **Crucially, do not ask a question that is already in the Interview History.**
- Maintain a professional and friendly tone.
- Return *only the single question text*, with no preamble, labels, or quotation marks.

Next Interview Question:
"""
    try:
        # Use the global LangChain llm object
        response = llm.invoke(prompt_text)
        question = response.content.strip()

        # Basic cleanup
        question = question.strip('"')
        if question.lower().startswith("next interview question:"):
             question = question[len("next interview question:"):].strip()

        # Ensure a question is returned
        if not question:
             raise ValueError("LLM returned empty question")
        # Prevent asking the same question twice in a row (simple check)
        if question_history and question.lower() == question_history[-1].lower():
             print("Warning: LLM generated the same question as the previous one. Using fallback.")
             raise ValueError("Repeated question")

        return question
    except Exception as e:
        print(f"Error generating question with LLM: {e}. Using fallback.")
        # Fallback question logic (same as before)
        fallback_options = [
            "Can you tell me about your experience with Python in a professional setting?",
            "Describe a project where you utilized machine learning frameworks like PyTorch or TensorFlow.",
            "What experience do you have with cloud platforms such as AWS, Azure, or GCP?",
            "Could you walk me through your approach to system design for a complex application?",
            "Tell me about a time you demonstrated leadership skills on a software project."
        ]
        # Simple fallback rotation, avoiding questions already asked
        available_fallbacks = [q for q in fallback_options if q not in question_history]
        if not available_fallbacks:
             return "Okay, let's move on. Can you summarize your key strengths for this role?" # Ultimate fallback
        return available_fallbacks[len(question_history) % len(available_fallbacks)]


@tool
def evaluate_interview_progress(response: str, num_questions_asked: int, question_history: list[str]) -> str:
    """
    Evaluates whether the interview should continue or end based on the candidate's latest response,
    the number of questions asked, and the overall interview context. Considers minimum (3) and maximum (7) question counts.

    Args:
        response (str): The candidate's latest response.
        num_questions_asked (int): The number of questions asked so far.
        question_history (list[str]): List of questions asked.

    Returns:
        str: Either 'continue' or 'end'.
    """
    min_questions = 3
    max_questions = 7

    # Basic rules first
    if num_questions_asked < min_questions:
        print(f"Evaluation: Continuing (Minimum {min_questions} questions not yet reached).")
        return "continue"
    if num_questions_asked >= max_questions:
        print(f"Evaluation: Ending (Maximum {max_questions} questions reached).")
        return "end"

    # LLM evaluation for intermediate stages
    prompt_text = f"""
You are an evaluation module for an AI interviewer. Your task is to decide whether to continue the interview based on the candidate's latest response and the interview progress.

Job Description (Reference):
{JOB_DESCRIPTION}

Interview Progress:
- Questions Asked So Far: {num_questions_asked} (Min: {min_questions}, Max: {max_questions})
- Question History: {', '.join(question_history)}

Candidate's Latest Response:
"{response}"

Evaluate the quality and relevance of the candidate's latest response in relation to the job description and typical expectations for a Senior Software Engineer.
Consider if enough information has been gathered or if more probing is needed.

Based on the progress and the quality of the latest response, should the interview continue to gather more information, or is it appropriate to end it now (either due to sufficient information, consistently poor responses, or reaching a natural conclusion point)?

Return *only* the single word 'continue' or 'end'.
"""
    try:
         # Use the global LangChain llm object
        llm_decision = llm.invoke(prompt_text).content.strip().lower()
        if llm_decision == "continue":
            print(f"Evaluation: LLM decided to continue.")
            return "continue"
        elif llm_decision == "end":
             print(f"Evaluation: LLM decided to end.")
             return "end"
        else:
             print(f"Warning: LLM evaluation returned unexpected value ('{llm_decision}'). Defaulting to continue.")
             return "continue" # Default to continue if LLM response is unclear
    except Exception as e:
        print(f"Error during LLM evaluation: {e}. Defaulting to continue.")
        return "continue" # Default to continue on error


# --- VAD and Recording Logic (Keep as is) ---

def calculate_rms(data):
    """Calculate Root Mean Square of audio data."""
    if data.size == 0:
        return 0
    return np.sqrt(np.mean(np.square(data.astype(np.float64))))

def vad_audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
    """Callback function for VAD during audio recording."""
    global audio_buffer, recording_active, silent_frames
    if not recording_active:
        return

    if status:
        print(f"Audio Status Warning: {status}", flush=True)

    audio_chunk = indata[:, 0] if CHANNELS > 1 else indata.flatten()
    audio_buffer.extend(audio_chunk)

    rms = calculate_rms(audio_chunk)
    # print(f"RMS: {rms:.2f}") # Uncomment for debugging

    if rms < SILENCE_THRESHOLD:
        silent_frames += 1
    else:
        silent_frames = 0

    if silent_frames >= silence_blocks_needed:
        if recording_active:
            print(f"Silence detected ({SILENCE_DURATION_SEC}s), stopping recording.", flush=True)
            recording_active = False

@tool
def record_with_vad(filename: str = "interview_response.wav") -> bool:
    """
    Records audio from the default input device with Voice Activity Detection (VAD).
    Stops recording after a period of silence or maximum duration. Saves the audio to a file.

    Args:
        filename (str): The name of the audio file to save. Will be stored in the recordings directory.

    Returns:
        bool: True if recording and saving were successful, False otherwise.
    """
    global audio_buffer, recording_active, silent_frames
    
    # Ensure we use absolute path in recordings directory
    filepath = os.path.join(RECORDINGS_DIR, os.path.basename(filename))
    
    # Create recordings directory if it doesn't exist (redundant but safe)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print(f"Will save recording to: {filepath}")
    
    audio_buffer.clear()
    recording_active = True
    silent_frames = 0
    start_time = time.time()

    print(f"\n Recording answer... Speak now.")
    print(f"(Will stop after {SILENCE_DURATION_SEC}s of silence or {MAX_RECORD_SEC}s max duration).", flush=True)

    # --- IMPORTANT: Select Input Device ---
    # Ensure this device index/name is the VIRTUAL CABLE OUTPUT where Zoom's audio goes.
    # Use sd.query_devices() to verify.
    devices = sd.query_devices() # Keep for debugging if needed
    print(devices)
            
    input_device = 1 # Example: Replace with index (e.g., 2) or name (e.g., 'BlackHole 2ch')
    # input_device = None # Use default input device unless specified

    # Check if default device is sufficient or if specific device needed
    if input_device is not None:
        print(f"Attempting to use specified input device: {input_device}")
    else:
        print("Using default system input device for recording.")
        try:
             default_input_device_info = sd.query_devices(kind='input')
             print(f"Default input device details: {default_input_device_info['name']}")
        except Exception as qe:
             print(f"Could not query default input device: {qe}")

    # --- End Device Selection ---

    stream = None
    try:
        stream = sd.InputStream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=vad_audio_callback,
            device=input_device # Use specified or default device
        )
        stream.start()
        print("Audio stream started. Listening...", flush=True)

        while recording_active:
            if time.time() - start_time > MAX_RECORD_SEC:
                print("\nMax recording time reached.", flush=True)
                recording_active = False
            sd.sleep(100) # Process audio events

        print("Recording loop finished.", flush=True)

    except sd.PortAudioError as pae:
        print(f"PortAudio Error during recording: {pae}", flush=True)
        print("Please check your audio device settings:")
        print(f"  - Is the correct INPUT device selected/default (should be virtual cable output)?")
        print(f"  - Is the sample rate ({SAMPLERATE}Hz) supported by the device?")
        print(f"  - Is the device in use by another application?")
        if input_device is not None:
             print(f"  - Specified device: {input_device}")
        sd.query_devices() # Print devices again for debugging
        if stream: stream.close()
        return False
    except Exception as e:
        print(f"An unexpected error occurred during recording: {e}", flush=True)
        if stream: stream.close()
        return False
    finally:
        if stream and stream.active:
            try:
                stream.stop()
                stream.close()
                print("Audio stream stopped and closed.", flush=True)
            except Exception as e_close:
                 print(f"Error closing stream: {e_close}", flush=True)

    # --- Save the recorded audio ---
    if not audio_buffer:
        print("Warning: No audio data was captured.", flush=True)
        return False

    audio_data = np.array(list(audio_buffer), dtype=np.int16)

    print(f"Saving audio to {filename} ({len(audio_data)/SAMPLERATE:.2f} seconds)...", flush=True)
    try:
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(np.dtype(DTYPE).itemsize)
            wf.setframerate(SAMPLERATE)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio successfully saved to {filepath}.", flush=True)
        return True
    except Exception as e:
        print(f" Error saving WAV file '{filepath}': {e}", flush=True)
        return False

@tool
def transcribe_audio(filename: str = "interview_response.wav") -> str | None:
    """
    Transcribes the audio file using the local whisper.cpp executable.

    Args:
        filename (str): Name of the audio file to transcribe. Will be looked up in recordings directory.

    Returns:
        str | None: The transcribed text if successful, None otherwise.
    """
    # Ensure we look in the recordings directory
    filepath = os.path.join(RECORDINGS_DIR, os.path.basename(filename))
    
    if not os.path.exists(filepath):
        print(f"Error: Audio file not found for transcription: {filepath}")
        return None
    if not os.path.exists(WHISPER_CPP_EXECUTABLE):
         print(f"Error: Whisper executable not found: {WHISPER_CPP_EXECUTABLE}")
         return None
    if not os.path.exists(WHISPER_MODEL_PATH):
        print(f" Error: Whisper model not found: {WHISPER_MODEL_PATH}")
        return None

    print(f"⏳ Transcribing {filename} with whisper.cpp...")
    command = [
        WHISPER_CPP_EXECUTABLE,
        "-m", WHISPER_MODEL_PATH,
        "-f", filepath,
        "--language", "en",
        "--output-txt", # Keep for debugging maybe
        # '--no-timestamps', # Add if you don't want timestamps in stdout
    ]

    try:
        # Increased timeout just in case
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
        transcription = result.stdout.strip()

        # --- Improved Cleaning Logic (Keep as is) ---
        lines = transcription.splitlines()
        cleaned_lines = []
        timestamp_pattern = re.compile(r'^\[\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}\]\s*')
        for line in lines:
            match = timestamp_pattern.match(line)
            if match:
                text_after_timestamp = line[match.end():].strip()
                cleaned_text = re.sub(r'\[.*?\]', '', text_after_timestamp).strip() # Remove bracketed sounds/music
                if cleaned_text:
                    cleaned_lines.append(cleaned_text)
            else:
                cleaned_text = re.sub(r'\[.*?\]', '', line).strip() # Remove bracketed sounds/music
                if cleaned_text:
                    cleaned_lines.append(cleaned_text)
        # --- End Improved Cleaning Logic ---

        cleaned_transcription = " ".join(cleaned_lines)

        if not cleaned_transcription:
             print(" Warning: Transcription resulted in empty text after cleaning.")
             # Depending on how the agent should react, return None or a specific string
             return "[No audible speech detected]" # Return indicator instead of None
        else:
             print(f"Transcription successful.")
             # print(f"   Transcript: '{cleaned_transcription}'") # Uncomment for debugging
             return cleaned_transcription

    except subprocess.CalledProcessError as e:
        print(f" Error during whisper.cpp execution (Code: {e.returncode}):")
        print("   STDOUT:", e.stdout)
        print("   STDERR:", e.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("Error: Transcription timed out.")
        return None
    except Exception as e:
        print(f" An unexpected error occurred during transcription: {e}")
        return None


# --- LangChain Agent Setup ---
tools = [
    join_zoom_meeting,
    speak_reply,
    generate_interview_question,
    record_with_vad,
    transcribe_audio,
    evaluate_interview_progress
]

# Pull ReAct agent prompt template
# This prompt is designed to work with create_react_agent and handles tool descriptions internally.
try:
    # Use the reliable prompt from the LangChain Hub
    prompt = hub.pull("hwchase17/react-chat")
    print(" Successfully pulled ReAct prompt from LangChain Hub.")
except Exception as hub_e:
     print(f" Warning: Could not pull prompt from hub ({hub_e}). Using a basic fallback structure.")
     # Define a basic prompt structure manually ONLY IF HUB PULL FAILS
     # This is less likely to be fully optimized for ReAct compared to the hub version.
     prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided tools to answer the user's request."),
            MessagesPlaceholder(variable_name="chat_history", optional=True), # If using memory
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Crucial for ReAct
        ])
# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)
print(" ReAct agent created.")

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # Keep verbose=True for debugging
    handle_parsing_errors=True # Helps manage minor LLM output formatting issues
)
print(" Agent Executor created.")

if __name__ == "__main__":
    print("🚀 Starting LangChain HR Interview Agent...")

    initial_task = f"""
    Conduct a technical interview for a Senior Software Engineer position using the available tools.

    Job Requirements:
    {JOB_DESCRIPTION}

   Interview Flow:
    1. Start by attempting to join the Zoom meeting using `join_zoom_meeting`.
    2. Introduce yourself briefly using the tool`speak_reply`. Say something like: "Hello, I'm the AI interviewer. Let's begin." Then proceed to the next step.
    3. Capture the candidate's name and the position they are applying for using the tool `record_with_vad`, saving the response as `interview_response.wav`. Then proceed to the next step.
    4. Transcribe the recorded answer using the tool `transcribe_audio`, passing the filename `interview_response.wav`. Then proceed to the next step.
    5. Speak a affirmative message showing you have heard it using the tool `speak_reply`. Say something like: "Thank you,  Let's start the interview.",  Then proceed to the next step.
    6. Generate the interview question using the tool `generate_interview_question`, keeping track of the asked questions in `question_history`. Then proceed to the next step.
    7. Speak the generated question using the tool `speak_reply`.
    8. Record the candidate's answer using the tool `record_with_vad`, saving it as `interview_response.wav`.
    9. Transcribe the recorded answer using the tool `transcribe_audio`, using the same filename `interview_response.wav`.
    10. Speak a affirmative message showing you have heard it using the tool `speak_reply`. Say something like: "Thank you, I appreciate your response.",  Then proceed to the next step.
    11. Repeat steps 6–10 until four questions have been asked.
    12. After the fourth question:
        a. Speak a closing statement using the tool `speak_reply`. Say something like: "Thank you for your time. That concludes the interview."
        b. Finish the process.

    Guidelines:
    - Ask exactly four questions.
    - Ensure you pass the candidate's transcribed response and the updated list of asked questions to  `generate_interview_question`.
    - If recording or transcription fails, state that you couldn't capture the response, and try asking different question again by repeating steps 6-10. Handle tool errors gracefully.
    - Start the interview now.

    """
    try:
        # Invoke the agent executor, providing ALL expected input variables
        result = agent_executor.invoke({
            "input": initial_task,
            "chat_history": [] 
            })
        print("\n Interview process finished.")
        print("\n=== Agent Final Output ===")
        print(result.get('output', "No final output captured."))

    except Exception as e:
        print(f"\n An error occurred during the agent execution: {e}")
        import traceback
        traceback.print_exc()
        print("\n Agent execution halted.")