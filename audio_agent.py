import sounddevice as sd
import numpy as np
import time

# Audio stream parameters
DURATION = 5  # Duration to record in seconds for each check
SAMPLE_RATE = 44100  # Standard audio sample rate

def audio_activity_analysis():
    print("Listening to classroom audio activity... (Speak or make noise)")
    
    # Record audio chunk from microphone
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    
    # Calculate Root Mean Square (RMS) energy to measure volume/activity level
    rms_energy = np.sqrt(np.mean(audio_data**2))
    
    # Simple classification logic based on volume threshold
    if rms_energy < 0.01:
        activity_status = "Silence / Low Engagement"
    elif rms_energy < 0.05:
        activity_status = "Active Classroom Discussion"
    else:
        activity_status = "Excess Background Noise / Distraction"
        
    print(f"[AUDIO ANALYSIS] RMS Energy: {rms_energy:.4f} -> Status: {activity_status}")
    return rms_energy, activity_status

if __name__ == "__main__":
    for i in range(3):  # Run 3 test iterations
        audio_activity_analysis()