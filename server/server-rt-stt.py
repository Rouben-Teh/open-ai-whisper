import json
from fastapi import FastAPI, UploadFile, File, WebSocket
from pydub import AudioSegment
import numpy as np
import io
from faster_whisper import WhisperModel
import uvicorn
import queue
import threading
import time

# ===== 参数 =====
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 2    
BLOCK_SECONDS = 0.5      
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_SECONDS)
CHUNK_FRAMES  = int(SAMPLE_RATE * CHUNK_SECONDS)

model = WhisperModel(
    "large-v3",
    device="cuda",      
    compute_type="int8"
)

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read the uploaded file
    audio_data = await file.read()
    
    # Load audio using pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    
    # Convert to mono, 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Get samples as numpy array, normalized to float32
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.sample_width == 2:  # 16-bit
        samples /= 32768.0
    elif audio.sample_width == 4:  # 32-bit
        samples /= 2147483648.0
    
    # Transcribe using Whisper
    segments, _ = model.transcribe(
        samples,
        language="zh",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False
    )
    
    # Combine all segments into text
    text = " ".join(segment.text for segment in segments)
    
    return {"text": text}

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    
    auto_cut = False
    cut = False
    silence_duration = 2.0
    last_audio_time = time.time()

    audio_buffer = []
    total_audio_seconds = 0.0
    transcribed_segments = []
    
    audio_queue = queue.Queue()
    message_queue = queue.Queue()  # Queue for sending messages from thread to async context
    stop_event = threading.Event()
    
    def transcriber():
        nonlocal auto_cut, cut, silence_duration, last_audio_time, audio_buffer, total_audio_seconds
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
                audio_buffer.append(data)
                
                total_frames = sum(len(b) for b in audio_buffer)
                
                if total_frames >= CHUNK_FRAMES:
                    total_audio_seconds += CHUNK_SECONDS
                    audio_data = np.concatenate(audio_buffer)[-CHUNK_FRAMES:]
                    audio_data = audio_data.flatten().astype(np.float32)
                    audio_buffer = []
                    
                    segments, _ = model.transcribe(
                        audio_data,
                        language="zh",
                        beam_size=1,
                        vad_filter=True,
                        condition_on_previous_text=False
                    )

                    segments = list(segments)

                    if len(segments) == 0:
                        print("Silence detected")
                        silence_duration += CHUNK_SECONDS
                    else:
                        print("Speech detected")
                        silence_duration = 0
                        cut = False

                        for s in segments:
                            print(s.start," ",s.end," ",s.text)
                            transcribed_segments.append(s.text)

                    # # 滑动窗口，保留最近 1 秒
                    # keep_frames = SAMPLE_RATE * 1
                    # audio_buffer = [audio_data[-keep_frames:]]

                if len(transcribed_segments) > 0 and auto_cut:
                    if(silence_duration >= 2.0 and not cut):
                        print("Auto cut triggered")
                        full_text = "".join(transcribed_segments)
                        # Put message in queue instead of calling async method directly
                        message_queue.put({
                            "type": "auto_cut",
                            "text": full_text
                        })
                        transcribed_segments.clear()
                        cut = True
                    
            except queue.Empty:
                continue
    
    transcriber_thread = threading.Thread(target=transcriber)
    transcriber_thread.start()
    
    try:
        while True:
            msg = await websocket.receive()

            # ===== 音频数据 =====
            if msg.get("bytes") is not None:
                audio_chunk = np.frombuffer(msg["bytes"], dtype=np.float32)
                audio_queue.put(audio_chunk)

            # ===== 正常结束信号 =====
            elif msg.get("text") is not None:
                text = msg["text"]

                # STOP
                if text == "STOP":
                    print("Received STOP signal")
                    break

                # JSON 消息
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "config":
                    auto_cut = bool(data.get("auto_cut", False))
                    print("auto_cut:", auto_cut)
            
            # ===== Check for messages from transcriber thread =====
            try:
                while True:
                    queued_msg = message_queue.get_nowait()
                    await websocket.send_json(queued_msg)
            except queue.Empty:
                pass

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        stop_event.set()
        transcriber_thread.join(timeout=5)
        full_text = "".join(transcribed_segments)

        try:
            await websocket.send_json({
                "type": "final",
                "text": full_text
            })
        except Exception:
            pass

        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)