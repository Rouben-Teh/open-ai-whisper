import os
import ffmpeg
import srt
from datetime import timedelta
from faster_whisper import WhisperModel

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

INPUT_FILE = "input.mp4"
OUTPUT_SRT = "output.srt"
TEMP_WAV = "temp_audio.wav"

# 1. 从视频/音频提取音轨
(
    ffmpeg
    .input(INPUT_FILE)
    .output(TEMP_WAV, ac=1, ar=16000)
    .overwrite_output()
    .run(quiet=True)
)

# 2. 加载 Whisper 模型
model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cpu",     # 没 GPU 改成 "cpu"
    compute_type="int8"
)

# 3. 语音识别
segments, info = model.transcribe(
    TEMP_WAV,
    language="zh",     # auto 可去掉
    beam_size=5
)

# 4. 生成 SRT
subtitles = []
for i, seg in enumerate(segments, start=1):
    subtitles.append(
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=seg.start),
            end=timedelta(seconds=seg.end),
            content=seg.text.strip()
        )
    )

# 5. 写入 SRT 文件
with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    f.write(srt.compose(subtitles))

# 6. 清理临时文件
os.remove(TEMP_WAV)

print("✅ 字幕生成完成:", OUTPUT_SRT)
