"""
Whisper transcription server — runs on the home PC, reached over Tailscale.

Takes over the free-text half of command recognition (song/album/playlist
names) from the Pi's local Vosk recognizer_full, which struggles with names
outside its training data. Control commands stay local on the Pi; see
assistant.py's is_name_bearing() / finalize() for what gets sent here.

Noise suppression lives here rather than on the Pi deliberately. The Pi runs
Vosk in real time and has no CPU to spare, while this only has to clean the
free-text audio that Whisper sees — which is the weak link, since the Pi's
restricted-grammar recogniser is already the noise-robust path.

Not tied to this project's Python environment — install and run on the home
PC with its own requirements.txt.
"""
import io
import os
import time
import wave

import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from faster_whisper import WhisperModel

try:
    import noisereduce as nr
except ImportError:
    nr = None

PORT       = 5051
MODEL_SIZE = "medium.en"  # was "small.en" — testing accuracy/speed tradeoff
AUTH_TOKEN = os.environ.get("WHISPER_AUTH_TOKEN", "")

# The Pi's mic noise is broadly stationary hiss once its 100Hz high-pass has
# taken out the mains hum, which is what spectral gating handles well.
# prop_decrease is deliberately below 1.0: gating hard enough to erase the
# noise also carves holes in the speech ("musical noise") and Whisper reads
# those artefacts as words. Lower this if names come back mangled.
DENOISE          = True
DENOISE_STRENGTH = 0.75

WHISPER_RATE = 16000

# Every clip the Pi sends is kept here, raw and denoised, so the pair can be
# compared by ear at http://<pc>:5051/ . Purely diagnostic; the Pi keeps its
# own copy of what it sent in data/command_recordings/.
SAVE_CLIPS = False   # debugging aid — set True to collect again
CLIPS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
MAX_CLIPS  = 30


def _decode_wav(data):
    """WAV bytes -> (float32 mono samples in -1..1, sample rate).

    Returns (None, None) for anything this can't read. This is a network
    boundary, so the payload is not assumed to be well-formed.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                return None, None
            rate = wf.getframerate()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    except Exception:
        return None, None
    return pcm.astype(np.float32) / 32768.0, rate


app = Flask(__name__)
print(f"Loading faster-whisper model '{MODEL_SIZE}' (CPU, int8)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
if DENOISE and nr is None:
    print("noisereduce not installed — transcribing raw audio. "
          "pip install noisereduce")
print(f"Model loaded. Denoise: {DENOISE and nr is not None}. Ready.")


def _write_wav(path, samples, rate):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def _save_clip(raw_samples, clean_samples, rate, text):
    """Keep the raw/denoised pair for listening back. Never raises: this is a
    diagnostic, and failing here would cost the Pi its Whisper transcription."""
    try:
        os.makedirs(CLIPS_DIR, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S")
        n = 1
        while os.path.exists(os.path.join(CLIPS_DIR, f"{stem}_raw.wav")):
            n += 1
            stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{n}"
        _write_wav(os.path.join(CLIPS_DIR, f"{stem}_raw.wav"), raw_samples, rate)
        if clean_samples is not None:
            _write_wav(os.path.join(CLIPS_DIR, f"{stem}_denoised.wav"), clean_samples, rate)
        with open(os.path.join(CLIPS_DIR, f"{stem}.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)

        stems = sorted({f.split("_raw")[0].split("_denoised")[0].rsplit(".", 1)[0]
                        for f in os.listdir(CLIPS_DIR)})
        for old in stems[:-MAX_CLIPS]:
            for suffix in ("_raw.wav", "_denoised.wav", ".txt"):
                try:
                    os.remove(os.path.join(CLIPS_DIR, old + suffix))
                except OSError:
                    pass
    except Exception as e:
        print(f"Could not save clip: {e}")


@app.route("/")
def index():
    """Browsable list of recent clips, raw next to denoised."""
    rows = []
    if os.path.isdir(CLIPS_DIR):
        stems = sorted({f.split("_raw")[0].split("_denoised")[0].rsplit(".", 1)[0]
                        for f in os.listdir(CLIPS_DIR) if f.endswith((".wav", ".txt"))},
                       reverse=True)
        for stem in stems:
            text = ""
            try:
                with open(os.path.join(CLIPS_DIR, stem + ".txt"), encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                pass
            clean = os.path.exists(os.path.join(CLIPS_DIR, stem + "_denoised.wav"))
            rows.append(f"""
              <tr><td>{stem}</td><td><b>{text or "&mdash;"}</b></td>
              <td>raw<br><audio controls preload=none src="/clips/{stem}_raw.wav"></audio></td>
              <td>{'denoised<br><audio controls preload=none src="/clips/' + stem + '_denoised.wav"></audio>' if clean else '&mdash;'}</td></tr>""")
    body = "".join(rows) or "<tr><td colspan=4>No clips yet — say a command with a song name.</td></tr>"
    return f"""<!doctype html><meta charset=utf-8><title>Whisper clips</title>
<style>body{{font-family:system-ui;margin:2rem;background:#111;color:#eee}}
table{{border-collapse:collapse}}td{{padding:.5rem .75rem;border-bottom:1px solid #333;vertical-align:top}}
b{{color:#6cf}}</style>
<h2>Clips received from the Pi</h2>
<p>Denoise: <b>{DENOISE and nr is not None}</b> &middot; strength {DENOISE_STRENGTH}
 &middot; saving clips: <b>{SAVE_CLIPS}</b>{"" if SAVE_CLIPS else " &mdash; set SAVE_CLIPS = True in transcribe_server.py to collect new ones"}</p>
<table>{body}</table>"""


@app.route("/clips/<name>")
def clip_file(name):
    if not name.endswith(".wav"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(CLIPS_DIR, name, mimetype="audio/wav")


def _denoise(audio, rate):
    """Returns denoised samples, or the originals if that isn't possible."""
    if nr is None:
        return audio
    return nr.reduce_noise(y=audio, sr=rate, stationary=True,
                           prop_decrease=DENOISE_STRENGTH)


@app.route("/denoise", methods=["POST"])
def denoise_preview():
    """Return the denoised audio itself, so it can be listened to.

    Same processing the transcription path applies, but handed back as a wav
    instead of text — the only way to hear what Whisper is actually being
    given, since the cleaned audio is otherwise discarded after transcription.
    """
    received = request.headers.get("X-Auth-Token")
    if AUTH_TOKEN and received != AUTH_TOKEN:
        return jsonify({"error": "unauthorized"}), 401

    audio, rate = _decode_wav(request.get_data())
    if audio is None or rate != WHISPER_RATE:
        return jsonify({"error": f"expected {WHISPER_RATE}Hz mono 16-bit wav"}), 400

    cleaned = _denoise(audio, rate)
    pcm = (np.clip(cleaned, -1.0, 1.0) * 32767).astype(np.int16)

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return Response(out.getvalue(), mimetype="audio/wav")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    received = request.headers.get("X-Auth-Token")
    if AUTH_TOKEN and received != AUTH_TOKEN:
        print(f"401: token mismatch (received {len(received) if received else 0} chars, "
              f"expected {len(AUTH_TOKEN)} chars)")
        return jsonify({"error": "unauthorized"}), 401

    raw = request.get_data()
    # ?denoise=0 transcribes the same clip untouched, so a recording can be
    # A/B'd against itself without restarting the server.
    want_denoise = DENOISE and request.args.get("denoise") != "0"

    audio, rate = _decode_wav(raw)
    denoise_ms = 0.0

    if audio is None:
        # Not PCM this can read — hand the bytes to faster-whisper and let it
        # decode them, which is what this server did before denoising existed.
        source = io.BytesIO(raw)
    elif rate != WHISPER_RATE:
        print(f"Unexpected sample rate {rate}, skipping denoise")
        source = io.BytesIO(raw)
    else:
        source = audio
        if want_denoise and nr is not None:
            started = time.monotonic()
            try:
                source = _denoise(audio, rate)
            except Exception as e:
                # Never fail the request over this: the Pi would fall back to
                # its much weaker local Vosk transcription.
                print(f"Denoise failed, using raw audio ({e})")
                source = audio
            denoise_ms = (time.monotonic() - started) * 1000

    started = time.monotonic()
    segments, _ = model.transcribe(
        source,
        language="en",
        beam_size=5,
        vad_filter=True,
        initial_prompt="Play, add, or search for a song, artist, album, or playlist.",
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    transcribe_ms = (time.monotonic() - started) * 1000

    # Timings matter: the Pi gives up after WHISPER_READ_TIMEOUT and falls
    # back to Vosk, so denoising has to stay small next to transcription.
    print(f"denoise {denoise_ms:.0f}ms + transcribe {transcribe_ms:.0f}ms -> {text!r}")

    if SAVE_CLIPS and audio is not None and rate == WHISPER_RATE:
        # source is the same object as audio when denoising was skipped or
        # failed, in which case there is no cleaned version worth saving.
        cleaned = source if isinstance(source, np.ndarray) and source is not audio else None
        _save_clip(audio, cleaned, rate, text)

    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
