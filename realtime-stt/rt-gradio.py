import gradio as gr
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.5
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_SECONDS)

audio_queue = queue.Queue()
running = False
segments_all = []

model = WhisperModel(
    "small",
    device="cuda",  
    compute_type="float32"
)

# ===== 录音回调 =====
def audio_callback(indata, frames, time_info, status):
    if running:
        audio_queue.put(indata.copy().reshape(-1))

def recorder():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=BLOCK_FRAMES
    ):
        while running:
            sd.sleep(10)

# ===== 主识别逻辑 =====
def start_stt():
    global running, segments_all
    running = True
    segments_all = []

    threading.Thread(target=recorder, daemon=True).start()

    buffer = np.zeros(0, dtype=np.float32)
    start_time = time.time()

    while running:
        try:
            chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        buffer = np.concatenate([buffer, chunk])

        if len(buffer) >= SAMPLE_RATE * 2:  #2s
            audio_input = buffer[-SAMPLE_RATE * 2:] #get last 2s in buffer

            segments, _ = model.transcribe(
                audio_input,
                language="zh",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )

            text_out = ""
            for s in segments:
                text_out += s.text + " "
                segments_all.append({
                    "start": s.start + (time.time() - start_time),
                    "end": s.end + (time.time() - start_time),
                    "text": s.text
                })

            yield text_out.strip()

def stop_stt():
    global running
    running = False

    srt_lines = []
    for i, s in enumerate(segments_all, 1):
        srt_lines.append(
            f"{i}\n"
            f"{format_time(s['start'])} --> {format_time(s['end'])}\n"
            f"{s['text']}\n"
        )

    srt_content = "\n".join(srt_lines)
    with open("output.srt", "w", encoding="utf-8") as f:
        f.write(srt_content)

    return "output.srt"

def format_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

# ===== Gradio UI =====
with gr.Blocks(title="🎙️ 实时语音字幕") as demo:
    gr.Markdown("## 🎙️ 实时语音识别 → SRT")

    output_text = gr.Textbox(label="实时字幕", lines=6)
    srt_file = gr.File(label="生成的 SRT")

    start_btn = gr.Button("▶️ 开始")
    stop_btn = gr.Button("⏹️ 停止并生成 SRT")

    start_btn.click(
        start_stt,
        outputs=output_text
    )

    stop_btn.click(
        stop_stt,
        outputs=srt_file
    )

demo.launch()
