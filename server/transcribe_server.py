"""
Whisper transcription server — runs on the home PC, reached over Tailscale.

Takes over the free-text half of command recognition (song/album/playlist
names) from the Pi's local Vosk recognizer_full, which struggles with names
outside its training data. Control commands stay local on the Pi; see
assistant.py's is_name_bearing() / finalize() for what gets sent here.

Not tied to this project's Python environment — install and run on the home
PC with its own requirements.txt.
"""
import io
import os

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

PORT       = 5051
MODEL_SIZE = "medium.en"  # was "small.en" — testing accuracy/speed tradeoff
AUTH_TOKEN = os.environ.get("WHISPER_AUTH_TOKEN", "")

app = Flask(__name__)
print(f"Loading faster-whisper model '{MODEL_SIZE}' (CPU, int8)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded. Ready.")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    received = request.headers.get("X-Auth-Token")
    if AUTH_TOKEN and received != AUTH_TOKEN:
        print(f"401: token mismatch (received {len(received) if received else 0} chars, "
              f"expected {len(AUTH_TOKEN)} chars)")
        return jsonify({"error": "unauthorized"}), 401

    audio = io.BytesIO(request.get_data())
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        initial_prompt="Play, add, or search for a song, artist, album, or playlist.",
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
