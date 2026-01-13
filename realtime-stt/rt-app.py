import sounddevice as sd
import numpy as np
import queue
import tempfile
import wave
from faster_whisper import WhisperModel
import os
import tempfile

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 3  # 每 3 秒识别一次

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def save_wav(filename, audio):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)

# === Whisper ===
model = WhisperModel(
    "medium",
    device="cuda",      
    compute_type="float16"
)

print("🎙️ Start real-time transcribe (ctrl+c Exit)")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
    callback=audio_callback
):
    buffer = b""
    try:
        while True:
            data = audio_queue.get()
            buffer += data.tobytes()

            if len(buffer) >= SAMPLE_RATE * 2 * CHUNK_SECONDS:
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)

                save_wav(path, buffer)

                segments, _ = model.transcribe(
                    path,
                    language="zh",
                    condition_on_previous_text=False
                )

                for s in segments:
                    print(s.text, end=" ", flush=True)

                os.remove(path)
                buffer = b""

    except KeyboardInterrupt:
        print("\n🛑 Stop")
