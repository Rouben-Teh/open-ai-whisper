import gradio as gr
import shutil
import os
from transcribe import transcribe_to_srt

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
# cuda_version = "12.9"
# cuda_bin_path = fr"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\{cuda_version}"

# # Add to PATH (so DLL loader can find it)
# os.environ["PATH"] +=  os.pathsep + cuda_bin_path

def process(file, language):
    if file is None:
        return None

    input_path = os.path.join(UPLOAD_DIR, os.path.basename(file))
    shutil.copy(file, input_path)

    srt_path = transcribe_to_srt(input_path, language)
    return srt_path

with gr.Blocks(title="Auto Subtitle Generator") as demo:
    gr.Markdown("## 🎬 音频 / 视频自动生成 SRT")

    language = gr.Dropdown(
        choices=["auto", "zh", "en"],
        value="auto",
        label="语言"
    )

    file_input = gr.File(
        label="上传音频或视频",
        file_types=["audio", "video"]
    )

    output_file = gr.File(label="生成的 SRT")

    btn = gr.Button("生成字幕")

    btn.click(
        fn=process,
        inputs=[file_input, language],
        outputs=output_file
    )

demo.launch(
    inbrowser=True,
    server_name="127.0.0.1",
    server_port=7860
)
