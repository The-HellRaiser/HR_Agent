import sounddevice as sd
import numpy as np
import wave
import subprocess
import time
import pyttsx3
import requests
import os
from zoomus import ZoomClient
import webbrowser
from urllib.parse import urlencode, urlparse, parse_qs

# Add Zoom credentials
ZOOM_API_KEY = "Uk0EINMjQ76NRkdPBBC5xw"
ZOOM_API_SECRET = "bLiKS6j9PjNcS7UbSEFsQ5A4iA411uJ5"

# Job Description Constants
JOB_DESCRIPTION = """
Senior Software Engineer Position:
- Lead development of AI-powered applications
- 5+ years of Python development experience
- Expert knowledge of machine learning frameworks (PyTorch, TensorFlow)
- Experience with cloud platforms (AWS/Azure/GCP)
- Strong system design and architecture skills
- Team leadership experience
"""

def parse_zoom_url(url):
    """Extract meeting ID and password from a Zoom URL"""
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')
    meeting_id = path_parts[-1]  # Get the last part of the path
    
    # Parse query parameters
    query_params = parse_qs(parsed.query)
    password = query_params.get('pwd', [None])[0]
    
    return meeting_id, password

def join_zoom_meeting(meeting_id, password=None):
    """Join a Zoom meeting using the default browser"""
    # Check if meeting_id is a full URL
    if meeting_id.startswith('http'):
        meeting_id, pwd = parse_zoom_url(meeting_id)
        if password is None:  # Only use URL password if none was explicitly provided
            password = pwd
    
    # Base Zoom join URL
    zoom_url = "zoommtg://zoom.us/join?"
    
    # Parameters for joining
    params = {
        'confno': meeting_id.replace(' ', ''),  # Remove spaces from meeting ID
    }
    
    if password:
        params['pwd'] = password
    
    # Construct full URL
    join_url = "https://us04web.zoom.us/j/5234862235?pwd=MnA4akU5S1lQMHZLZlI4eU5IbFUrQT09"
    
    # Open URL in default browser
    webbrowser.open(join_url)
    print(f"Joining Zoom meeting: {meeting_id}")
    
    # Wait for Zoom to launch
    time.sleep(5)

# Initialize Zoom client
client = ZoomClient(ZOOM_API_KEY, ZOOM_API_SECRET,api_account_id="cEXnlrthTEeW19zqKJt_WA")

def speak_reply(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    
    # Set the output device to the virtual audio cable
    # Find the device index of your virtual cable.  The following code may help:
    devices = sd.query_devices()
    print(devices)
    # Then, replace 'device_index' with the correct index.
    
    # Example for macOS with BlackHole (you might need to adjust the device index)
    engine.setProperty('MacBook Pro Microphone', 1)  # Replace device_index with the correct index
    
    engine.say(text)
    engine.runAndWait()

# Audio buffer
samplerate = 16000  # Whisper operates at 16kHz
audio_buffer = []
def generate_and_speak_response(user_text):
    url = "http://localhost:11434/api/generate"
    
    # Request payload for Ollama
    payload = {
        "model": "phi3.5:3.8b-mini-instruct-q5_K_M",  # or another model you have pulled
        "prompt": user_text,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes
        reply = response.json()["response"]
        print("Bot:", reply)
        
        # Speak the response in Zoom
        speak_reply(reply)
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return None

def generate_interview_question(previous_response=None):
    """Generate contextual interview questions"""
    prompt = f"""
    Job Description: {JOB_DESCRIPTION}
    Previous Response: {previous_response if previous_response else 'None'}
    
    Generate a natural, conversational interview question that:
    1. Is relevant to the job requirements
    2. Follows up on the candidate's previous response if available
    3. Maintains a professional but friendly tone
    
    Return only the question, no additional text.
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3.5:3.8b-mini-instruct-q5_K_M",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return "Could you tell me more about your experience?"

def start_interview():
    """Initialize the interview with a greeting"""
    greeting = "Hello! I'm your AI interviewer today. Thank you for joining. Let's start with your background in Python development."
    speak_reply(greeting)
    time.sleep(2)  # Short pause after greeting

# Audio recording callback
def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer.extend(indata.copy())

# Capture Zoom Audio (16-bit WAV format)
def record_audio(filename, duration=5):
    global audio_buffer
    audio_buffer = []  # Reset buffer
    with sd.InputStream(callback=audio_callback, samplerate=samplerate, channels=1, dtype="int16"):
        time.sleep(duration)  # Record for duration
    audio_data = np.array(audio_buffer, dtype=np.int16)

    # Save to WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

# Transcribe using whisper.cpp
def transcribe_audio(filename):
    model_path = "/Users/researchassistant/HR Agent/HR_Agent/whisper.cpp/models/ggml-base.en.bin"

    result = subprocess.run(
        ["./whisper.cpp/build/bin/whisper-cli", "-m", model_path, "-f", filename, "--language", "en"],
        capture_output=True, text=True
    )

    return result.stdout.strip()

# Modified main loop with Zoom joining
if __name__ == "__main__":
    # Ask for meeting URL or ID
    meeting_input = "5234862235" 
    meeting_password = "970483" 
    
    # Join the meeting
    join_zoom_meeting(meeting_input, meeting_password)
    
    # Wait for meeting to connect
    time.sleep(5)
    
    # Start interview
    start_interview()
    previous_response = None
    
    # Start the main audio processing loop
    while True:
        print("🎤 Recording Zoom audio...")
        record_audio("zoom_audio.wav", duration=20)  # Extended to 20 seconds
        
        print("📝 Transcribing...")
        response = transcribe_audio("zoom_audio.wav")
        if response:
            print("Candidate:", response)
            previous_response = response
            
            # Generate and ask next question
            next_question = generate_interview_question(previous_response)
            print("AI Interviewer:", next_question)
            speak_reply(next_question)
            
            # Wait between questions
            time.sleep(10)

