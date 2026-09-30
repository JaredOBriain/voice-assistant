"""
Whisper transcription server - runs on the home PC, reached over Tailscale.

Takes over the free-text half of command recognition (song/album/playlist
names) from the Pi's local Vosk recognizer_full, which struggles with names
outside its training data. Control commands stay local on the Pi; see
assistant.py's is_name_bearing() / finalize() for what gets sent here.

Noise suppression lives here rather than on the Pi deliberately. The Pi runs
Vosk in real time and has no CPU to spare, while this only has to clean the
free-text audio that Whisper sees - which is the weak link, since the Pi's
restricted-grammar recogniser is already the noise-robust path.

Not tied to this project's Python environment - install and run on the home
PC with its own requirements.txt.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
from faster_whisper import WhisperModel

try:
    import noisereduce as nr
    _nr_import_error = None
except ImportError as e:
    # Keep the reason. "Not installed" is only one cause - it is just as often
    # installed against a different interpreter, or installed but unable to
    # import because one of ITS dependencies is missing.
    nr = None
    _nr_import_error = e

PORT       = 5051
# small.en was tried and rejected. It was genuinely 3x faster (10.5s -> 3.5s)
# and looked fine on a six-clip bench, but in real use it needed repeating far
# too often. The bench was misleading because it held almost no long song
# titles - which is the entire reason this server exists, and exactly where a
# bigger model earns its keep. Don't re-run that comparison and conclude
# small.en is fine; it isn't, on the audio that matters.
#
# Accuracy is the point here and ~11s is the accepted price. Whisper always
# encodes a padded 30-second window, so a 3s command costs the same as a 25s
# one - clip length is not a lever, only model size and hardware are.
MODEL_SIZE = "medium.en"
AUTH_TOKEN = os.environ.get("WHISPER_AUTH_TOKEN", "")

# The Pi's mic noise is broadly stationary hiss once its 100Hz high-pass has
# taken out the mains hum, which is what spectral gating handles well.
# prop_decrease is deliberately below 1.0: gating hard enough to erase the
# noise also carves holes in the speech ("musical noise") and Whisper reads
# those artefacts as words. Lower this if names come back mangled.
DENOISE          = True
DENOISE_STRENGTH = 0.75

WHISPER_RATE = 16000

# Which engine transcribes. "faster-whisper" is the shipped one; "constme"
# shells out to Const-me/Whisper, which runs on Direct3D 11 compute shaders
# and so can use this PC's RX 590 - faster-whisper cannot, because CTranslate2
# has no native ROCm and the community forks start at gfx900 while Polaris is
# gfx803. Measured ~3x faster at equivalent accuracy; this switch exists to
# test that properly before committing to it.
#
# Set to "constme" on this branch so ordinary driving exercises it and the
# accuracy can be judged from real use rather than six clips. faster-whisper
# stays loaded as the fallback. Override per request with ?engine=... to put
# the same clip through both without restarting anything.
ENGINE = "constme"

CONSTME_EXE   = os.environ.get("CONSTME_EXE", r"C:\constme\main.exe")
# No default: which GGML model to use is a real choice (size, .en or not) and
# guessing a path here would silently transcribe with something other than
# what was intended. Set it explicitly:
#   $env:CONSTME_MODEL = "C:\path\to\ggml-medium.en.bin"
CONSTME_MODEL = os.environ.get("CONSTME_MODEL", "")

# Greedy decoding (1) was measured at only 7% faster than 5 - 11.34s to 10.50s
# - because beam search touches the decoder and a spoken command emits a
# handful of tokens. The cost is the encoder. So dropping the beam gives up
# accuracy for almost nothing, and it is back at 5.
BEAM_SIZE = 5

# Whisper conditions on this text, so it biases decoding toward the words the
# assistant actually accepts. Worth listing the control words and not just the
# music ones: short utterances carry little context and are where it guesses
# worst - "pause" came back as "All of a sudden" without them.
INITIAL_PROMPT = (
    "Voice commands for a music assistant. "
    "Play, add, or search for a song, artist, album, playlist, or podcast. "
    "Pause. Resume. Skip. Next track. Previous track. Go back. Stop the music. "
    "Loop. Repeat. Break. Volume. Bluetooth. Headphones. Speaker. "
    "Resume podcast. Yes. No."
)

# Every clip the Pi sends is kept here, raw and denoised, so the pair can be
# compared by ear at http://<pc>:5051/ . Purely diagnostic; the Pi keeps its
# own copy of what it sent in data/command_recordings/.
SAVE_CLIPS = True    # ON for engine testing; set False when done
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
    print(f"Denoising OFF - could not import noisereduce: {_nr_import_error}")
    print(f"  running interpreter: {sys.executable}")
    print(f'  install into THIS interpreter: "{sys.executable}" -m pip install noisereduce')
# A Const-me that cannot start falls back per request, which is right for
# keeping the Pi working but means a whole week of "testing Const-me" could
# quietly be faster-whisper. Say so at startup, where it will be noticed.
if ENGINE == "constme":
    problems = []
    if not CONSTME_MODEL:
        problems.append("CONSTME_MODEL is not set")
    elif not os.path.exists(CONSTME_MODEL):
        problems.append(f"model not found: {CONSTME_MODEL}")
    if not os.path.exists(CONSTME_EXE):
        problems.append(f"exe not found: {CONSTME_EXE}")
    if problems:
        print("\n*** ENGINE is 'constme' but it cannot run: ***")
        for p in problems:
            print(f"      {p}")
        print("    Every request will fall back to faster-whisper, so you would")
        print("    be testing the wrong engine. Fix before collecting results.\n")
    else:
        print(f"Engine: constme  ({CONSTME_EXE}, {os.path.basename(CONSTME_MODEL)})")
else:
    print(f"Engine: {ENGINE}")

print(f"Model loaded. Denoise: {DENOISE and nr is not None}. Ready.")


def _write_wav(path, samples, rate):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def _save_clip(raw_samples, clean_samples, rate, text,
               engine="", transcribe_ms=0.0, denoise_ms=0.0):
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
        # Which engine produced this, and what it cost. Without it the saved
        # clips are useless for comparing engines - you cannot tell afterwards
        # which one wrote the text.
        with open(os.path.join(CLIPS_DIR, f"{stem}.json"), "w", encoding="utf-8") as fh:
            json.dump({"engine": engine, "text": text,
                       "transcribe_ms": round(transcribe_ms),
                       "denoise_ms": round(denoise_ms),
                       "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

        stems = sorted({f.split("_raw")[0].split("_denoised")[0].rsplit(".", 1)[0]
                        for f in os.listdir(CLIPS_DIR)})
        for old in stems[:-MAX_CLIPS]:
            for suffix in ("_raw.wav", "_denoised.wav", ".txt", ".json"):
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
            text, meta = "", {}
            try:
                with open(os.path.join(CLIPS_DIR, stem + ".json"), encoding="utf-8") as fh:
                    meta = json.load(fh)
                text = meta.get("text", "")
            except Exception:
                try:
                    with open(os.path.join(CLIPS_DIR, stem + ".txt"), encoding="utf-8") as fh:
                        text = fh.read()
                except OSError:
                    pass
            badge = (f"{meta.get('engine','')} {meta.get('transcribe_ms','')}ms"
                     if meta.get("engine") else "")
            clean = os.path.exists(os.path.join(CLIPS_DIR, stem + "_denoised.wav"))
            rows.append(f"""
              <tr><td>{stem}<br><small>{badge}</small></td><td><b>{text or "&mdash;"}</b></td>
              <td>raw<br><audio controls preload=none src="/clips/{stem}_raw.wav"></audio></td>
              <td>{'denoised<br><audio controls preload=none src="/clips/' + stem + '_denoised.wav"></audio>' if clean else '&mdash;'}</td></tr>""")
    body = "".join(rows) or "<tr><td colspan=4>No clips yet - say a command with a song name.</td></tr>"
    return f"""<!doctype html><meta charset=utf-8><title>Whisper clips</title>
<style>body{{font-family:system-ui;margin:2rem;background:#111;color:#eee}}
table{{border-collapse:collapse}}td{{padding:.5rem .75rem;border-bottom:1px solid #333;vertical-align:top}}
b{{color:#6cf}}</style>
<h2>Clips received from the Pi</h2>
<p>Engine: <b>{ENGINE}</b> &middot; Denoise: <b>{DENOISE and nr is not None}</b> &middot; strength {DENOISE_STRENGTH}
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
    instead of text - the only way to hear what Whisper is actually being
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


def _transcribe_faster_whisper(source):
    segments, _ = model.transcribe(
        source,
        language="en",
        beam_size=BEAM_SIZE,
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def _transcribe_constme(samples, rate):
    """Run Const-me/Whisper over the samples. Returns None if it can't.

    It is a one-shot CLI with no daemon mode, but that costs nothing here:
    ggml models are memory-mapped, so repeat invocations measured the same as
    the first. Both engines get INITIAL_PROMPT so the comparison is fair.
    """
    if not CONSTME_MODEL:
        print("CONSTME_MODEL is not set - point it at a GGML model, e.g. "
              '$env:CONSTME_MODEL = "C:\\constme\\ggml-medium.en.bin"')
        return None
    if not (os.path.exists(CONSTME_EXE) and os.path.exists(CONSTME_MODEL)):
        print(f"Const-me not found (exe={CONSTME_EXE!r} model={CONSTME_MODEL!r})")
        return None

    tmpdir = tempfile.mkdtemp()
    wav_path = os.path.join(tmpdir, "clip.wav")
    try:
        _write_wav(wav_path, samples, rate)
        cmd = [CONSTME_EXE, "-m", CONSTME_MODEL, "-f", wav_path,
               "-l", "en", "-nt", "-otxt", "--prompt", INITIAL_PROMPT]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # -otxt writes beside the input; the name differs between builds.
        # utf-8-sig because Const-me emits a BOM, and a stray U+FEFF makes
        # every later text comparison fail for no visible reason.
        for candidate in (wav_path + ".txt", os.path.splitext(wav_path)[0] + ".txt"):
            if os.path.exists(candidate):
                text = open(candidate, encoding="utf-8-sig", errors="replace").read()
                return " ".join(text.split()).lstrip("\ufeff")

        text = " ".join(proc.stdout.split()).lstrip("\ufeff")
        if not text:
            print(f"Const-me produced nothing (rc={proc.returncode}): "
                  f"{proc.stderr.strip()[:160]}")
            return None
        return text
    except Exception as e:
        print(f"Const-me failed: {e}")
        return None
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


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
        # Not PCM this can read - hand the bytes to faster-whisper and let it
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

    # ?engine= lets the same clip go through both without a restart, which is
    # the point of the switch. Const-me needs real samples at WHISPER_RATE, so
    # the odd payloads that fall back to BytesIO stay on faster-whisper.
    engine = (request.args.get("engine") or ENGINE).lower()
    if engine == "constme" and not isinstance(source, np.ndarray):
        print("Const-me needs decoded samples; using faster-whisper for this one")
        engine = "faster-whisper"

    started = time.monotonic()
    if engine == "constme":
        text = _transcribe_constme(source, rate)
        if text is None:
            # Never fail the request over the experimental engine: the Pi
            # would drop to its much weaker local Vosk transcription.
            print("Const-me unavailable, falling back to faster-whisper")
            engine = "faster-whisper (constme failed)"
            text = _transcribe_faster_whisper(source)
    else:
        engine = "faster-whisper"
        text = _transcribe_faster_whisper(source)
    transcribe_ms = (time.monotonic() - started) * 1000

    # Timings matter: the Pi gives up after WHISPER_READ_TIMEOUT and falls
    # back to Vosk, so denoising has to stay small next to transcription.
    print(f"[{engine}] denoise {denoise_ms:.0f}ms + transcribe "
          f"{transcribe_ms:.0f}ms -> {text!r}")

    if SAVE_CLIPS and audio is not None and rate == WHISPER_RATE:
        # source is the same object as audio when denoising was skipped or
        # failed, in which case there is no cleaned version worth saving.
        cleaned = source if isinstance(source, np.ndarray) and source is not audio else None
        _save_clip(audio, cleaned, rate, text,
                   engine=engine, transcribe_ms=transcribe_ms, denoise_ms=denoise_ms)

    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
