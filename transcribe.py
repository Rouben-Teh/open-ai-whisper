import os
import ffmpeg
import srt
from datetime import timedelta
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8"
)

def transcribe_to_srt(input_file: str, language: str) -> str:
    temp_wav = "temp_audio.wav"
    output_srt = input_file + ".srt"

    # 1. 提取音频
    (
        ffmpeg
        .input(input_file)
        .output(temp_wav, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    lang = None if language == "auto" else language

    # 2. 识别
    segments, info = model.transcribe(
        temp_wav,
        language=lang
    )

    # 3. 生成 SRT
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

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    os.remove(temp_wav)
    return output_srt
