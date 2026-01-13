import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

# ===== 参数 =====
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 2    
BLOCK_SECONDS = 0.5      
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_SECONDS)
CHUNK_FRAMES  = int(SAMPLE_RATE * CHUNK_SECONDS)

audio_queue = queue.Queue()
audio_buffer = []

model = WhisperModel(
    "large-v3",
    device="cuda",      
    compute_type="int8"
)

# ===== 音频回调 =====
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy().reshape(-1))

def recorder():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        # dtype="int16",
        callback=audio_callback,
        blocksize=CHUNK_FRAMES
    ):
        print("🎙️ 实时 STT 开始（Ctrl+C 停止）")

        while(True):
            sd.sleep(10)

# ===== Whisper =====
def transcriber():
    global audio_buffer
    try:
        while True:
            data = audio_queue.get()
            audio_buffer.append(data)

            total_frames = sum(len(b) for b in audio_buffer)

            if total_frames >= CHUNK_FRAMES:
                audio_data = np.concatenate(audio_buffer)[-CHUNK_FRAMES:]
                # audio_buffer = []
                audio_data = audio_data.flatten().astype(np.float32)

                segments, _ = model.transcribe(
                    audio_data,
                    language="zh",
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False
                )

                for s in segments:
                    print(s.text, end=" ", flush=True)

                # 滑动窗口，保留最近 1 秒
                keep_frames = SAMPLE_RATE * 1
                audio_buffer = [audio_data[-keep_frames:]]

    except KeyboardInterrupt:
        print("\n🛑 已停止")

threading.Thread(target=recorder, daemon=True).start()
transcriber()