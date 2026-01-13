from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    "Zh_7_prompt.wav",
    task="translate"
)

for s in segments:
    print(s.text)