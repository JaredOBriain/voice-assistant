import ollama
import subprocess
import os
import re
import time
import signal
import json
import wave
import io
import pyaudio
import threading
import sys
import requests
import numpy as np
from scipy.signal import butter, sosfilt
from collections import deque
from queue import Queue, Empty
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# All paths are relative to the project root, so this repo works no matter
# where it's cloned to. Only YTDLP_PATH stays absolute since it's a
# system-wide tool installed outside the project.
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "piper", "en_US-lessac-medium.onnx")
CONFIG_PATH     = os.path.join(BASE_DIR, "piper", "en_US-lessac-medium.onnx.json")
OUTPUT_WAV      = os.path.join(BASE_DIR, "data", "test.wav")
THINKING_SOUND  = os.path.join(BASE_DIR, "assets", "process.wav")
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-en-us-0.15")
BEEP_SOUND      = os.path.join(BASE_DIR, "assets", "beep.wav")
PIPER_PATH      = os.path.join(BASE_DIR, "piper", "piper")
MUSIC_FOLDER    = os.path.join(BASE_DIR, "data", "Music")
PLAYLIST_FILE   = os.path.join(BASE_DIR, "data", "playlists.json")
YTDLP_PATH      = "/home/jpie/.local/bin/yt-dlp"

# yt-dlp needs a JavaScript runtime to solve YouTube's signature challenges.
# Without one it silently falls back to clients like visionos/m3u8 whose URLs
# mostly answer "HTTP Error 403: Forbidden", so songs fail to download for no
# visible reason. Only deno is enabled by default and it isn't packaged for
# Debian; node is, but at 20.x it's below yt-dlp's required 22.0.0. quickjs
# is in Debian main (2025.04.26, min is 2023.12.09) and is the lightest
# option, so it's what the Pi uses: sudo apt install quickjs
YTDLP_JS_RUNTIME = ["--js-runtimes", "quickjs"]

# ---------------------------------------------------------------------------
# openWakeWord (fully offline, open source, no account needed)
# ---------------------------------------------------------------------------
OWW_MODEL_NAME = "hey_jarvis"   # pretrained: hey_jarvis, alexa, hey_mycroft
WAKE_THRESHOLD = 0.5            # 0-1; lower = more sensitive, higher = fewer false triggers

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
MIC_DEVICE_INDEX = 1
MIC_RATE         = 48000
VOSK_RATE        = 16000
RESAMPLE_FACTOR  = MIC_RATE // VOSK_RATE
WAKE_CHUNK_16K   = 1280   # openWakeWord's frame size, at VOSK_RATE
MAX_QUEUE        = 30

# ---------------------------------------------------------------------------
# Bluetooth
# ---------------------------------------------------------------------------
BT_SPEAKER_SINK = "bluez_output.78_66_F3_2B_B7_E2.1"

BT_HEADPHONE_MAC = "A8:F5:E1:6A:ED:64"

TRUST_MIC_SOURCE = "alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono"

using_bluetooth = False

# ---------------------------------------------------------------------------
# Remote transcription (home PC, over Tailscale)
# ---------------------------------------------------------------------------
# Offloads only the free-text half of command recognition (song/album/
# playlist names — see is_name_bearing()) to a faster-whisper server on the
# home PC. Local Vosk is the automatic fallback if this is unreachable or
# too slow; see finalize() in listen_for_command().
WHISPER_SERVER_URL      = "http://100.71.69.96:5051/transcribe"  # Tailscale IP, update after setup
WHISPER_AUTH_TOKEN      = os.environ.get("WHISPER_AUTH_TOKEN", "")  # from .env via the service; never hardcode it
WHISPER_CONNECT_TIMEOUT = 3   # fail fast if unreachable (no signal / PC off)
WHISPER_READ_TIMEOUT    = 20  # medium.en on CPU int8 takes ~15s; small.en fit in 10s

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
music_lock = threading.Lock()

# Set while speak_with_piper() is actively playing audio, so the wake-word
# listener can ignore the assistant's own voice coming back through the mic
# (there's no acoustic echo cancellation in this pipeline).
assistant_speaking = threading.Event()

vosk_model           = None
recognizer           = None
recognizer_full      = None
oww_model            = None
oww_wakeword_key     = None
oww_blank_feature_buffer = None
is_listening         = False
wake_word_detected   = False
conversation_history = []
script_running       = True

music_process = None
queue         = deque(maxlen=MAX_QUEUE)
history       = deque(maxlen=MAX_QUEUE)
current_song  = None
is_paused     = False
is_looping    = False
music_thread  = None
is_listening_for_command = False
current_volume           = 20


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def create_beep_sound():
    os.makedirs(os.path.dirname(BEEP_SOUND), exist_ok=True)
    if os.path.exists(BEEP_SOUND):
        return
    print("Creating beep sound...")
    try:
        import math, struct
        SAMPLE_RATE = 44100
        DURATION    = 0.3
        FREQUENCY   = 880
        with wave.open(BEEP_SOUND, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            for i in range(int(SAMPLE_RATE * DURATION)):
                fade = 1.0
                if i < SAMPLE_RATE * 0.1:
                    fade = i / (SAMPLE_RATE * 0.1)
                elif i > SAMPLE_RATE * (DURATION - 0.1):
                    fade = (SAMPLE_RATE * DURATION - i) / (SAMPLE_RATE * 0.1)
                sample = int(fade * 0.5 * math.sin(2 * math.pi * FREQUENCY * i / SAMPLE_RATE) * 32767)
                f.writeframes(struct.pack('<h', sample))
        print(f"Beep created at {BEEP_SOUND}")
    except Exception as e:
        print(f"Could not create beep: {e}")


# Command vocabulary — biases recognition toward these phrases.
# "[unk]" keeps song/album names open for free speech.
VOSK_GRAMMAR = json.dumps([
    "play", "play me", "put on", "i want to hear", "i want to listen to",
    "add", "add to queue",
    "album", "play album", "add album",
    "playlist",
    "stop music", "stop the music", "stop playing", "pause music", "turn off music",
    "pause", "hold on", "wait",
    "resume", "continue", "unpause", "play on", "carry on",
    "skip", "next song", "next track",
    "back", "previous song", "previous track", "go back",
    "loop", "repeat", "loop this",
    "break", "stop looping", "stop repeating",
    "bluetooth", "headphones", "switch to bluetooth", "switch to headphones", "use headphones",
    "speaker", "built in", "switch to speaker", "use speaker", "switch to built in",
    "exit", "quit", "stop", "goodbye",
    "start over", "clear history", "forget that", "reset",
    "[unk]"
])


def initialize_vosk():
    global vosk_model, recognizer, recognizer_full
    from vosk import Model, KaldiRecognizer
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"Vosk model not found at {VOSK_MODEL_PATH}")
        sys.exit(1)
    vosk_model = Model(VOSK_MODEL_PATH)
    # Restricted recognizer: noise-robust command-word detection
    recognizer = KaldiRecognizer(vosk_model, VOSK_RATE, VOSK_GRAMMAR)
    # Full-vocabulary recognizer: captures free-text song/album/playlist names
    recognizer_full = KaldiRecognizer(vosk_model, VOSK_RATE)
    print("Vosk loaded (restricted + full recognizers).")


NAME_BEARING_PREFIXES = (
    "play album", "add album", "album",
    "add to queue", "add",
    "playlist",
    "play me", "play", "put on",
    "i want to hear", "i want to listen to",
)


def is_name_bearing(text):
    return any(text.startswith(prefix) for prefix in NAME_BEARING_PREFIXES)


def choose_command_text(restricted, full):
    """Pick the best transcription. The restricted recognizer reliably detects
    the command word even in noise. If that command carries a free-text name
    (play/add/album/playlist), use the full-vocabulary transcription so the
    name isn't swallowed as [unk]."""
    restricted = (restricted or "").strip()
    full       = (full or "").strip()
    if not restricted:
        return full
    if is_name_bearing(restricted):
        return full if full else restricted
    return restricted


def strip_punctuation(text):
    """Whisper punctuates and capitalises what it hears ("Play, Bohemian
    Rhapsody."), but every matcher downstream works on bare words — an
    inserted comma alone is enough to stop "play, x" matching the "play "
    prefix, so the command falls through to "Sorry, I didn't catch that".
    Apostrophes survive because song titles genuinely contain them
    ("Livin' On A Prayer")."""
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def transcribe_remote(pcm_bytes):
    """Send buffered 16kHz mono int16 audio to the home PC's faster-whisper
    server. Returns the transcribed text, or None if unreachable/failed —
    callers should fall back to the local Vosk transcription in that case."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(VOSK_RATE)
        wf.writeframes(pcm_bytes)

    try:
        response = requests.post(
            WHISPER_SERVER_URL,
            data=wav_buffer.getvalue(),
            headers={"X-Auth-Token": WHISPER_AUTH_TOKEN},
            timeout=(WHISPER_CONNECT_TIMEOUT, WHISPER_READ_TIMEOUT),
        )
        response.raise_for_status()
        text = strip_punctuation(response.json().get("text", ""))
        print(f"Remote transcription: {text!r}")
        return text or None
    except Exception as e:
        print(f"Remote transcription unavailable, using local result ({e})")
        return None


def initialize_openwakeword():
    global oww_model, oww_wakeword_key, oww_blank_feature_buffer
    from openwakeword.model import Model
    import openwakeword, glob

    # openwakeword>=0.4 ships pretrained models inside the package itself
    # (no download step) and Model() now takes explicit file paths rather
    # than model names. It keys predictions by filename-minus-extension
    # (e.g. "hey_jarvis_v0.1"), not the bare OWW_MODEL_NAME, so that key
    # has to be derived from the resolved path rather than assumed.
    resources_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    matches = glob.glob(os.path.join(resources_dir, f"{OWW_MODEL_NAME}_*.onnx"))
    if not matches:
        print(f"openWakeWord model files for '{OWW_MODEL_NAME}' not found in {resources_dir}")
        sys.exit(1)

    try:
        oww_model = Model(wakeword_model_paths=[matches[0]])
        oww_wakeword_key = os.path.basename(matches[0])[:-len(".onnx")]
        # Model.reset() (openwakeword 0.4.0) only clears the prediction
        # smoothing buffer — it does NOT clear preprocessor.raw_data_buffer,
        # .melspectrogram_buffer or .feature_buffer, which hold up to ~10s of
        # audio/embedding history (see openwakeword/utils.py AudioFeatures).
        # Caching a blank feature buffer here lets reset_wake_word_state()
        # restore that history to empty cheaply, without recomputing it.
        oww_blank_feature_buffer = oww_model.preprocessor._get_embeddings(
            np.zeros(160000).astype(np.int16)
        )
        print(f"openWakeWord loaded. Wake word: '{OWW_MODEL_NAME.replace('_', ' ')}'")
    except Exception as e:
        print(f"openWakeWord init failed: {e}")
        sys.exit(1)


def reset_wake_word_state():
    """Fully clear openWakeWord's state between listening sessions.

    oww_model.reset() alone leaves ~10s of stale raw audio / melspectrogram /
    embedding history sitting in oww_model.preprocessor. That history still
    contains the embeddings for the "hey jarvis" utterance that just got
    detected, so the very next listen_for_wake_word() call could see a high
    score again within its first few chunks — a second, spurious detection
    from the same utterance, not a real new one. This mirrors what
    AudioFeatures.__init__ sets up, so the preprocessor starts genuinely
    blank instead of just "unscored"."""
    oww_model.reset()
    pp = oww_model.preprocessor
    pp.raw_data_buffer.clear()
    pp.accumulated_samples   = 0
    pp.melspectrogram_buffer = np.ones((76, 32))
    pp.feature_buffer        = oww_blank_feature_buffer.copy()


# Recorded commands measured at roughly -22 dBFS peak with a 7-9 dB SNR, and
# the noise was dominated by sub-100Hz rumble: mains hum at 50Hz sat ~37 dB
# above the noise median, with harmonics at 100 and 150Hz. That rumble carries
# no speech but sets the peak level, which is why speech ended up so quiet.
# So: high-pass first, then boost — boosting first would just amplify the hum
# and clip on it.
HIGHPASS_HZ = 100   # 4th order, so 50Hz hum lands about 24 dB down
MIC_GAIN    = 12.0  # raw speech peaks ~1330/32768, so x12 lands near -6 dBFS.
                    # The old code applied x2 here, so this is 6x louder than before.

_highpass_sos   = butter(4, HIGHPASS_HZ, "highpass", fs=MIC_RATE, output="sos")
_highpass_state = np.zeros((_highpass_sos.shape[0], 2))


def reset_mic_filter():
    """Clear the high-pass state between captures, so one command's tail can't
    ring into the start of the next."""
    global _highpass_state
    _highpass_state = np.zeros((_highpass_sos.shape[0], 2))


def prepare_mic_audio(data):
    """High-pass, boost, then decimate a chunk of mic audio to Vosk's rate.

    The filter state carries across calls on purpose: this runs per 8000-sample
    chunk, and restarting the filter each time would put a discontinuity into
    the signal every 167ms."""
    global _highpass_state
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    audio, _highpass_state = sosfilt(_highpass_sos, audio, zi=_highpass_state)
    audio = np.clip(audio * MIC_GAIN, -32768, 32767).astype(np.int16)
    return audio[::RESAMPLE_FACTOR].tobytes() if RESAMPLE_FACTOR != 1 else audio.tobytes()


# Measured on real captures after the high-pass and MIC_GAIN above: silence
# sits at 640-715 mean-abs and speech at 1160+, while ambient noise alone
# reached 787. **Retune this whenever HIGHPASS_HZ or MIC_GAIN changes** — it
# was 400 for the unfiltered x2 signal, and leaving it there would have made
# every silence look like speech, so no command would ever end before timeout.
SILENCE_THRESHOLD = 900

# How long the speaker must stay quiet before a command is considered finished.
# This is now the only thing that ends a command (besides the overall timeout),
# so lower it if replies feel sluggish — at the cost of clipping slow speech.
SILENCE_END_SECONDS = 1.5


def is_silent(data, threshold=SILENCE_THRESHOLD):
    return np.abs(np.frombuffer(data, dtype=np.int16)).mean() < threshold


def start_thinking_sound():
    return subprocess.Popen(
        ["aplay", "-q", THINKING_SOUND],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def stop_thinking_sound(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=0.3)
        except:
            proc.kill()


def play_beep():
    try:
        if os.path.exists(BEEP_SOUND):
            subprocess.run(
                ["aplay", "-q", BEEP_SOUND],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            print("\a", end='', flush=True)
    except Exception as e:
        print(f"Beep error: {e}")


def speak_with_piper(text):
    assistant_speaking.set()
    try:
        try:
            subprocess.run(
                [PIPER_PATH, "-m", MODEL_PATH, "-c", CONFIG_PATH, "-f", OUTPUT_WAV],
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Piper error: {e}")
            return
        if os.path.exists(OUTPUT_WAV):
            subprocess.run(["aplay", OUTPUT_WAV], stderr=subprocess.DEVNULL)
        else:
            print("Piper did not produce output.")
    finally:
        # Small grace period: cabin reverb/echo tail can outlast the audio
        # itself, so don't start listening again the instant playback ends.
        time.sleep(0.4)
        assistant_speaking.clear()


# ---------------------------------------------------------------------------
# Mic reader thread
# ---------------------------------------------------------------------------
# Reading the mic inline between Vosk calls dropped ~20% of every command.
# Vosk saturates the core for ~65ms per chunk, PipeWire's client thread isn't
# serviced in time, and those samples are gone — silently, because the reads
# pass exception_on_overflow=False. Measured 79.7% capture inline against
# 100% with a thread that does nothing but read. That missing fifth chopped
# syllables out of the middle of commands, which no amount of gain or
# denoising downstream can recover.
AUDIO_CHUNK_FRAMES = WAKE_CHUNK_16K * RESAMPLE_FACTOR  # one wake-word chunk
AUDIO_QUEUE_MAX    = 40                                # ~3.2s of 48kHz audio

_audio_queue    = Queue(maxsize=AUDIO_QUEUE_MAX)
_audio_leftover = bytearray()
_reader_stop    = threading.Event()
_reader_thread  = None


def _audio_reader(stream):
    while not _reader_stop.is_set():
        try:
            chunk = stream.read(AUDIO_CHUNK_FRAMES, exception_on_overflow=False)
        except Exception as e:
            print(f"Audio reader error: {e}")
            time.sleep(0.1)
            continue
        if _audio_queue.full():
            # Nothing is listening right now (TTS, a download, playback), so
            # drop the oldest chunk rather than grow without bound. The old
            # code discarded this backlog too, just via PortAudio overruns.
            try:
                _audio_queue.get_nowait()
            except Empty:
                pass
        _audio_queue.put(chunk)


def start_audio_reader(stream):
    global _reader_thread
    _reader_stop.clear()
    _reader_thread = threading.Thread(target=_audio_reader, args=(stream,), daemon=True)
    _reader_thread.start()


def stop_audio_reader():
    _reader_stop.set()


def drain_audio():
    """Discard buffered audio so a listen starts from now, not from whatever
    piled up while the assistant was busy."""
    _audio_leftover.clear()
    while True:
        try:
            _audio_queue.get_nowait()
        except Empty:
            return


def read_audio(frames):
    """Blocking read of `frames` 48kHz frames, assembled from the reader.

    Returns short only when shutting down, which the callers' `while
    is_listening` guards already handle."""
    want = frames * 2  # int16
    while len(_audio_leftover) < want:
        try:
            _audio_leftover.extend(_audio_queue.get(timeout=1))
        except Empty:
            if _reader_stop.is_set() or not is_listening:
                break
    out = bytes(_audio_leftover[:want])
    del _audio_leftover[:want]
    return out


def open_mic_stream(p, frames_per_buffer=8000):
    return p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_RATE,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=frames_per_buffer
    )


# ---------------------------------------------------------------------------
# Music engine
# ---------------------------------------------------------------------------

def _kill_current_process():
    global music_process
    if music_process and music_process.poll() is None:
        music_process.terminate()
        try:
            music_process.wait(timeout=1)
        except:
            music_process.kill()
    music_process = None


def _playback_loop():
    global music_process, current_song, is_paused, is_looping
    while True:
        with music_lock:
            if is_looping and current_song:
                filepath = current_song
            elif queue:
                filepath     = queue.popleft()
                current_song = filepath
                is_paused    = False
            else:
                current_song = None
                print("Queue finished.")
                return

        print(f"Now playing: {os.path.splitext(os.path.basename(filepath))[0]}")
        try:
            with music_lock:
                music_process = subprocess.Popen(
                    ["mpg123", "-q", filepath],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            proc = music_process
            proc.wait()
            with music_lock:
                if proc.returncode == 0 and not is_looping:
                    history.append(filepath)
        except FileNotFoundError:
            print("mpg123 not installed. Run: sudo apt install mpg123")
            speak_with_piper("mpg123 is not installed.")
            return
        except Exception as e:
            print(f"Playback error: {e}")


def _ensure_playback_thread():
    global music_thread
    if music_thread is None or not music_thread.is_alive():
        music_thread = threading.Thread(target=_playback_loop, daemon=True)
        music_thread.start()


def stop_music():
    global current_song, is_paused, is_looping
    with music_lock:
        queue.clear()
        current_song = None
        is_paused    = False
        is_looping   = False
        _kill_current_process()
    print("Music stopped.")


def pause_music():
    global is_paused
    with music_lock:
        if music_process and music_process.poll() is None and not is_paused:
            music_process.send_signal(signal.SIGSTOP)
            is_paused = True
            return True
    return False


def resume_music():
    global is_paused
    with music_lock:
        if music_process and music_process.poll() is None and is_paused:
            music_process.send_signal(signal.SIGCONT)
            is_paused = False
            return True
    return False


def skip_song():
    global is_looping
    with music_lock:
        if current_song:
            history.append(current_song)
        is_looping = False
        _kill_current_process()


def back_song():
    with music_lock:
        if not history:
            return False
        prev = history.pop()
        if current_song:
            queue.appendleft(current_song)
        queue.appendleft(prev)
        _kill_current_process()
    return True


def loop_song():
    global is_looping
    with music_lock:
        if not current_song:
            return False
        is_looping = True
    return True


def break_loop():
    global is_looping
    with music_lock:
        is_looping = False


def play_now(filepath):
    global is_looping
    with music_lock:
        is_looping = False
        if current_song:
            queue.appendleft(current_song)
        queue.appendleft(filepath)
        _kill_current_process()
    _ensure_playback_thread()


def enqueue_song(filepath):
    with music_lock:
        if len(queue) >= MAX_QUEUE:
            return False
        queue.append(filepath)
    _ensure_playback_thread()
    return True


# ---------------------------------------------------------------------------
# Playlist management
# ---------------------------------------------------------------------------

def _playlist_track_path(stored):
    """Map a stored playlist entry onto the current MUSIC_FOLDER.

    Playlists store absolute paths, so moving the project (songs used to live
    in ~/Music) left every entry pointing at a file that no longer exists —
    each playlist silently reported itself as empty. Tracks only ever live
    directly in MUSIC_FOLDER, so the filename is the real identity of an
    entry and the directory part can be rebuilt from wherever the project is
    checked out now."""
    return os.path.join(MUSIC_FOLDER, os.path.basename(stored))


def load_playlists():
    if not os.path.exists(PLAYLIST_FILE):
        return {}
    try:
        with open(PLAYLIST_FILE, 'r') as f:
            playlists = json.load(f)
    except Exception as e:
        print(f"Playlist load error: {e}")
        return {}
    # Rewritten on every load, so the next save_playlists() quietly migrates
    # the file to the current location.
    return {
        name: [_playlist_track_path(track) for track in tracks]
        for name, tracks in playlists.items()
    }


def save_playlists(playlists):
    try:
        with open(PLAYLIST_FILE, 'w') as f:
            json.dump(playlists, f, indent=2)
    except Exception as e:
        print(f"Playlist save error: {e}")


def play_playlist(name):
    playlists = load_playlists()
    name_lower = name.lower().strip()
    match = next((k for k in playlists if k.lower() == name_lower), None)
    if not match:
        speak_with_piper(f"I couldn't find a playlist called {name}.")
        return
    songs = playlists[match]
    if not songs:
        speak_with_piper(f"The playlist {match} is empty.")
        return
    for filepath in songs:
        if os.path.exists(filepath):
            enqueue_song(filepath)
    speak_with_piper(f"Playing playlist {match}, {len(songs)} songs.")


# ---------------------------------------------------------------------------
# Song finding / downloading
# ---------------------------------------------------------------------------

def _words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def find_song_in_library(song_name):
    os.makedirs(MUSIC_FOLDER, exist_ok=True)
    needle_words = _words(song_name)
    if not needle_words:
        return None
    for filename in os.listdir(MUSIC_FOLDER):
        if not filename.lower().endswith(".mp3"):
            continue
        hay_words = _words(os.path.splitext(filename)[0])
        # Whole-word match only — a raw substring check let short mis-heard
        # fragments (e.g. "plan") match unrelated words that merely contain
        # them (e.g. "airplane").
        if needle_words <= hay_words or hay_words <= needle_words:
            print(f"[SONG] found in library: {filename}", flush=True)
            return os.path.join(MUSIC_FOLDER, filename)
    print(f"[SONG] not in library: {song_name}", flush=True)
    return None


def download_from_youtube(song_name):
    print(f"[YTDLP] path check: {YTDLP_PATH} exists={os.path.exists(YTDLP_PATH)}", flush=True)
    print(f"[YTDLP] searching: {song_name}", flush=True)
    output_template = os.path.join(MUSIC_FOLDER, "%(title)s.%(ext)s")
    try:
        result = subprocess.run(
            [
                YTDLP_PATH,
                *YTDLP_JS_RUNTIME,
                "--extract-audio", "--audio-format", "mp3", "--audio-quality", "0",
                "--output", output_template,
                "--no-playlist", "--match-filter", "duration < 600",
                f"ytsearch1:{song_name} official audio"
            ],
            capture_output=True, text=True, timeout=120
        )
        print(f"[YTDLP] return code: {result.returncode}", flush=True)
        if result.stdout:
            print(f"[YTDLP] stdout (last 500 chars): {result.stdout[-500:]}", flush=True)
        if result.returncode != 0:
            print(f"[YTDLP] stderr: {result.stderr}", flush=True)
            return None
        mp3s = [os.path.join(MUSIC_FOLDER, f) for f in os.listdir(MUSIC_FOLDER) if f.endswith(".mp3")]
        if not mp3s:
            print("[YTDLP] no mp3 files found in folder after download", flush=True)
            return None
        latest = max(mp3s, key=os.path.getmtime)
        print(f"[YTDLP] downloaded: {os.path.basename(latest)}", flush=True)
        return latest
    except subprocess.TimeoutExpired:
        print("[YTDLP] timed out after 120s", flush=True)
    except FileNotFoundError:
        print(f"[YTDLP] binary not found at {YTDLP_PATH}", flush=True)
    except Exception as e:
        print(f"[YTDLP] unexpected error: {e}", flush=True)
    return None


def resolve_song(song_name):
    print(f"[SONG] resolving: '{song_name}'", flush=True)
    result = find_song_in_library(song_name) or download_from_youtube(song_name)
    print(f"[SONG] resolved to: {result}", flush=True)
    return result


def get_album_tracklist(album_name):
    print(f"Getting tracklist: {album_name}")
    try:
        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a music database. Respond with ONLY a JSON array of track titles "
                        "in order. Example: [\"Track One\", \"Track Two\"]. No explanation, no markdown."
                    )
                },
                {"role": "user", "content": f"List all tracks on: {album_name}"}
            ]
        )
        content = response["message"]["content"].strip().replace("```json", "").replace("```", "").strip()
        tracklist = json.loads(content)
        if isinstance(tracklist, list) and tracklist:
            print(f"{len(tracklist)} tracks found for {album_name}")
            return tracklist
    except Exception as e:
        print(f"Tracklist error: {e}")
    return []


def find_album_in_library(album_name):
    if not os.path.exists(MUSIC_FOLDER):
        return []
    needle_words = _words(album_name)
    if not needle_words:
        return []
    matches = []
    for filename in os.listdir(MUSIC_FOLDER):
        if not filename.lower().endswith(".mp3"):
            continue
        hay_words = _words(filename)
        if needle_words <= hay_words:
            matches.append(os.path.join(MUSIC_FOLDER, filename))
    return sorted(matches)


# ---------------------------------------------------------------------------
# Command detection
# ---------------------------------------------------------------------------

def extract_song_name(command):
    for prefix in [
        "add to queue ", "add ",
        "play me ", "play ", "put on ",
        "i want to hear ", "i want to listen to "
    ]:
        if command.lower().startswith(prefix):
            return command[len(prefix):].strip()
    return command.strip()


def extract_album_name(command):
    for prefix in ["play album ", "add album ", "album "]:
        if command.lower().startswith(prefix):
            return command[len(prefix):].strip()
    return None


def is_play_command(cmd):
    return any(cmd.startswith(t) for t in [
        "play ", "play me ", "put on ", "i want to hear ", "i want to listen to "
    ]) and not cmd.startswith("play album ")


def is_add_command(cmd):
    return any(cmd.startswith(t) for t in ["add ", "add to queue "]) \
           and not cmd.startswith("add album ")


def is_album_command(cmd):
    return any(cmd.startswith(t) for t in ["album ", "play album ", "add album "])


def is_stop_music_command(cmd):
    return any(p in cmd for p in [
        "stop music", "stop the music", "stop playing", "pause music", "turn off music"
    ])


def is_pause_command(cmd):
    return any(p in cmd for p in ["pause", "hold on", "wait"])


def is_resume_command(cmd):
    return any(p in cmd for p in ["resume", "continue", "unpause", "play on", "carry on"])


def is_skip_command(cmd):
    return any(p in cmd for p in ["skip", "next song", "next track"])


def is_back_command(cmd):
    return any(p in cmd for p in ["back", "previous song", "previous track", "go back"])


def is_loop_command(cmd):
    return any(p in cmd for p in ["loop", "repeat", "loop this"])


def is_break_command(cmd):
    return any(p in cmd for p in ["break", "stop looping", "stop repeating"])


def is_playlist_command(cmd):
    return cmd.startswith("playlist ")


def extract_playlist_name(cmd):
    return cmd[len("playlist "):].strip()


def is_bt_command(cmd):
    return any(p in cmd for p in [
        "bluetooth", "headphones", "switch to bluetooth",
        "switch to headphones", "use headphones"
    ])


def is_speaker_command(cmd):
    return any(p in cmd for p in [
        "speaker", "built in", "built-in",
        "switch to speaker", "use speaker", "switch to built in"
    ])


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def handle_play_command(command):
    song_name = extract_song_name(command)
    if not song_name:
        speak_with_piper("What song would you like me to play?")
        return
    print(f"Looking for: {song_name}")
    filepath = resolve_song(song_name)
    if filepath:
        speak_with_piper(f"Playing {os.path.splitext(os.path.basename(filepath))[0]}.")
        play_now(filepath)
    else:
        speak_with_piper(f"Sorry, I couldn't find {song_name}.")


def handle_add_command(command):
    song_name = extract_song_name(command)
    if not song_name:
        speak_with_piper("What song would you like me to add?")
        return
    print(f"Adding: {song_name}")
    speak_with_piper(f"Looking for {song_name}.")
    filepath = resolve_song(song_name)
    if filepath:
        if enqueue_song(filepath):
            with music_lock:
                pos = len(queue)
            speak_with_piper(f"Added {os.path.splitext(os.path.basename(filepath))[0]} at position {pos}.")
        else:
            speak_with_piper("The queue is full.")
    else:
        speak_with_piper(f"Sorry, I couldn't find {song_name}.")


def handle_album_command(command):
    album_name = extract_album_name(command)
    if not album_name:
        speak_with_piper("What album would you like to play?")
        return

    library_tracks = find_album_in_library(album_name)
    if library_tracks:
        speak_with_piper(f"Found {len(library_tracks)} tracks. Adding to the queue.")
        added = sum(1 for t in library_tracks if enqueue_song(t))
        speak_with_piper(f"Added {added} songs from {album_name}.")
        return

    speak_with_piper(f"I don't have {album_name}. Getting the tracklist now.")
    tracklist = get_album_tracklist(album_name)
    if not tracklist:
        speak_with_piper(f"Sorry, I couldn't get the tracklist for {album_name}.")
        return

    speak_with_piper(f"Found {len(tracklist)} tracks. Downloading now, this may take a few minutes.")
    downloaded = 0
    for i, track in enumerate(tracklist):
        search_term = f"{track} {album_name}"
        print(f"Downloading {i+1}/{len(tracklist)}: {search_term}")
        filepath = download_from_youtube(search_term)
        if filepath:
            enqueue_song(filepath)
            downloaded += 1
        else:
            print(f"Failed: {track}")

    if downloaded:
        speak_with_piper(f"Downloaded {downloaded} tracks from {album_name}.")
    else:
        speak_with_piper(f"Sorry, I couldn't download any tracks from {album_name}.")


def handle_pause():
    if pause_music():
        speak_with_piper("Paused.")
    else:
        speak_with_piper("Nothing is playing.")


def handle_resume():
    if resume_music():
        speak_with_piper("Resuming.")
    else:
        speak_with_piper("Nothing is paused.")


def handle_skip():
    with music_lock:
        has_next    = len(queue) > 0
        has_current = current_song is not None
    if not has_current:
        speak_with_piper("Nothing is playing.")
        return
    speak_with_piper("Skipping.")
    skip_song()
    if not has_next:
        speak_with_piper("That was the last song.")


def handle_back():
    with music_lock:
        has_history = len(history) > 0
    if not has_history:
        speak_with_piper("There is no previous song.")
        return
    speak_with_piper("Going back.")
    back_song()
    _ensure_playback_thread()


def handle_loop():
    if loop_song():
        with music_lock:
            song = current_song
        if song:
            speak_with_piper(f"Looping {os.path.splitext(os.path.basename(song))[0]}. Say break to stop.")
    else:
        speak_with_piper("Nothing is playing to loop.")


def handle_break():
    break_loop()
    speak_with_piper("Loop stopped.")


# ---------------------------------------------------------------------------
# Bluetooth audio switching
# ---------------------------------------------------------------------------

def find_bt_mic_source():
    """Resolve the headset's PipeWire source by MAC, or None if absent.

    The node name is not stable across reboots — it has appeared as both
    `bluez_input.A8_F5_E1_6A_ED_64.0` and `bluez_input.A8:F5:E1:6A:ED:64`
    on this exact hardware, so a hardcoded name silently stops matching and
    the mic switch fails. Matching on the address covers both spellings.
    Returning None doubles as the "headset isn't connected" answer, since
    the source only exists while it is."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=5
        )
    except Exception as e:
        print(f"Could not list audio sources: {e}")
        return None

    wanted = (BT_HEADPHONE_MAC, BT_HEADPHONE_MAC.replace(":", "_"))
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        name = fields[1] if len(fields) > 1 else ""
        if name.startswith("bluez_input.") and any(mac in name for mac in wanted):
            return name
    return None


def set_startup_audio_defaults():
    global using_bluetooth
    print("Setting startup audio defaults...")
    try:
        subprocess.run(["pactl", "set-default-sink", BT_SPEAKER_SINK],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # The headset mic is clearly better than the USB one, so prefer it
        # whenever the headset is actually connected. If it isn't, its source
        # doesn't exist and we fall back rather than leaving the assistant deaf.
        bt_mic = find_bt_mic_source()
        source = bt_mic or TRUST_MIC_SOURCE
        subprocess.run(["pactl", "set-default-source", source],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        using_bluetooth = bt_mic is not None
        print(f"Sink:   {BT_SPEAKER_SINK}")
        print(f"Source: {source}")
    except Exception as e:
        print(f"Startup audio error: {e}")


def switch_to_headphones():
    global using_bluetooth

    if using_bluetooth:
        switch_to_speaker()
        return

    bt_mic = find_bt_mic_source()
    if not bt_mic:
        speak_with_piper("Bluetooth headphones are not connected.")
        return

    try:
        # Deliberately no set-card-profile here. The headset's profile is
        # pinned to HFP by ~/.config/wireplumber/wireplumber.conf.d/
        # 51-shokz-hfp.conf at connect time, and switching it afterwards does
        # not bring the mic back anyway — see switch_to_speaker().
        subprocess.run(
            ["pactl", "set-default-source", bt_mic],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        using_bluetooth = True
        print(f"Switched mic to Bluetooth headset ({bt_mic}).")
        speak_with_piper("Switched to headset microphone.")

    except subprocess.CalledProcessError as e:
        print(f"BT mic switch error: {e}")
        speak_with_piper("Failed to switch to headset microphone.")


def switch_to_speaker():
    global using_bluetooth

    try:
        # Only the default source changes. This used to also flip the headset
        # card to a2dp-sink, which was doubly wrong: the call failed silently
        # whenever A2DP wasn't on offer (stranding the headset in 16kHz), and
        # when it did succeed it destroyed the HFP mic node — which does not
        # come back on switching the profile again, only on a reconnect.
        subprocess.run(
            ["pactl", "set-default-source", TRUST_MIC_SOURCE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        using_bluetooth = False
        print("Reverted to USB microphone.")
        speak_with_piper("Switched back to USB microphone.")

    except Exception as e:
        print(f"Mic revert error: {e}")


# ---------------------------------------------------------------------------
# Command recordings (diagnostics)
# ---------------------------------------------------------------------------

COMMAND_RECORDINGS_DIR = os.path.join(BASE_DIR, "data", "command_recordings")
MAX_COMMAND_RECORDINGS = 30


def save_command_recording(pcm_bytes, decoded):
    """Keep the audio a command was decoded from, next to what each recogniser
    made of it, so a misheard command can actually be listened back to.

    This is the same 16kHz mono PCM that `transcribe_remote()` wraps and posts
    to the Whisper server, so what's saved is what the server heard. Commands
    that produced nothing are saved too — "it didn't hear me at all" is the
    case most worth having a recording of. Served by /recordings."""
    if not pcm_bytes:
        return
    try:
        os.makedirs(COMMAND_RECORDINGS_DIR, exist_ok=True)

        # finalize() can run more than once inside a single command capture
        # (it returns early on an empty result and the loop carries on), so
        # same-second names do collide in practice.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base  = os.path.join(COMMAND_RECORDINGS_DIR, stamp)
        n = 1
        while os.path.exists(f"{base}.wav"):
            n += 1
            base = os.path.join(COMMAND_RECORDINGS_DIR, f"{stamp}_{n}")

        with wave.open(f"{base}.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(VOSK_RATE)
            wf.writeframes(pcm_bytes)

        with open(f"{base}.json", "w") as f:
            json.dump({
                "recorded":   time.strftime("%Y-%m-%d %H:%M:%S"),
                "seconds":    round(len(pcm_bytes) / (VOSK_RATE * 2), 2),
                "restricted": decoded.get("restricted", ""),
                "full":       decoded.get("full", ""),
                "remote":     decoded.get("remote"),
                "final":      decoded.get("final", ""),
            }, f, indent=2)

        stems = sorted({os.path.splitext(f)[0] for f in os.listdir(COMMAND_RECORDINGS_DIR)})
        for old in stems[:-MAX_COMMAND_RECORDINGS]:
            for ext in (".wav", ".json"):
                try:
                    os.remove(os.path.join(COMMAND_RECORDINGS_DIR, old + ext))
                except OSError:
                    pass
    except Exception as e:
        print(f"Could not save command recording: {e}")


# ---------------------------------------------------------------------------
# Voice recognition
# ---------------------------------------------------------------------------

def listen_for_command(timeout_seconds=10):
    # Audio comes from the reader thread, never from the stream directly —
    # see _audio_reader(). The mic also can't be opened twice at once, so
    # there is exactly one stream and one reader for the whole process.
    drain_audio()

    print("\nListening for command...")

    start_time      = time.time()
    silence_start   = None
    speech_detected = False
    final_text      = ""
    audio_buffer    = bytearray()

    recognizer.Reset()
    recognizer_full.Reset()
    reset_mic_filter()

    decoded = {}  # what each recogniser made of the audio, for the recording
    # Vosk finalises a segment whenever AcceptWaveform() returns True, which it
    # does on any pause mid-phrase. FinalResult() then only returns the last
    # segment, so segments are collected as they close and joined at the end.
    parts_r, parts_f = [], []

    def finalize():
        parts_r.append(json.loads(recognizer.FinalResult()).get("text", "").strip())
        parts_f.append(json.loads(recognizer_full.FinalResult()).get("text", "").strip())
        r = " ".join(p for p in parts_r if p).strip()
        f = " ".join(p for p in parts_f if p).strip()
        remote = None
        if is_name_bearing(r) or is_name_bearing(f):
            remote = transcribe_remote(bytes(audio_buffer))
            if remote:
                f = remote
        chosen = choose_command_text(r, f)
        decoded.update(restricted=r, full=f, remote=remote, final=chosen)
        return chosen

    while is_listening:
        if time.time() - start_time > timeout_seconds:
            print("Timeout.")
            final_text = finalize()
            break

        raw = read_audio(8000)
        if not raw:
            break

        # Every chunk is fed to both recognizers regardless of is_silent() —
        # ambient noise here (~200-350) sits too close to actual speech level
        # for amplitude alone to safely gate what reaches Vosk. is_silent()
        # is used only below, to decide when to start/reset the "how long
        # since we last heard speech" timer that ends the command.
        data = prepare_mic_audio(raw)
        audio_buffer.extend(data)

        # Deliberately no break here. Vosk's endpointing fires on short pauses
        # inside a phrase, and breaking on it truncated commands: every capture
        # measured ended with under 0.7s of trailing silence, never reaching the
        # silence timer below. Collect the closed segment and keep listening —
        # only silence or the timeout ends a command now.
        if recognizer.AcceptWaveform(data):
            segment = json.loads(recognizer.Result()).get("text", "").strip()
            if segment:
                parts_r.append(segment)
                speech_detected = True
        if recognizer_full.AcceptWaveform(data):
            segment = json.loads(recognizer_full.Result()).get("text", "").strip()
            if segment:
                parts_f.append(segment)
                speech_detected = True

        partial = json.loads(recognizer_full.PartialResult()).get("partial", "")
        if partial:
            speech_detected = True

        # is_silent() must see the SAME audio SILENCE_THRESHOLD was measured
        # against, which is the processed signal — not `raw`. Raw sits around
        # 50-570 mean-abs, permanently under the threshold, so testing it made
        # every chunk look silent and cut the speaker off 1.5s after they
        # started talking.
        if is_silent(data):
            if speech_detected:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_END_SECONDS:
                    final_text = finalize()
                    if final_text:
                        print(f"Command: {final_text}")
                    break
        else:
            silence_start = None
            if partial:
                print(f"Hearing: {partial}    ", end='\r')

    save_command_recording(bytes(audio_buffer), decoded)
    return final_text.strip()


def open_wake_word_stream():
    """Opens the wake-word mic stream once, for the life of the process.
    Previously this was opened and torn down on every single wake cycle
    (every ~10-30s all day); repeatedly rebuilding the audio pipeline is a
    likely source of startup artifacts (pop/glitch/buffer-priming) right
    after each reopen, which lined up with spurious near-instant "wake word
    detected" triggers regardless of what was actually happening acoustically
    (confirmed: it still happened with no music playing, at a consistent
    ~1s delay every time — see data/debug_wakes/). Keeping one persistent
    stream removes that per-cycle restart entirely."""
    read_size = WAKE_CHUNK_16K * RESAMPLE_FACTOR
    p      = pyaudio.PyAudio()
    stream = open_mic_stream(p, frames_per_buffer=read_size)
    stream.start_stream()
    return p, stream, read_size


def listen_for_wake_word(read_size):
    global wake_word_detected

    # The reader thread keeps filling the queue the whole time we're off
    # handling a command, speaking or playing music. Drop that backlog so we
    # don't process stale audio as if it had just been spoken.
    drain_audio()

    reset_wake_word_state()

    print(f"\nListening for wake word: '{OWW_MODEL_NAME.replace('_', ' ')}'")

    while is_listening:
        if wake_word_detected:
            break
        try:
            raw = read_audio(read_size)
            if not raw:
                continue

            if assistant_speaking.is_set():
                # Drain the stream but don't let the assistant hear itself.
                continue

            pcm = np.frombuffer(raw, dtype=np.int16)[::RESAMPLE_FACTOR]
            prediction = oww_model.predict(pcm)
            score = prediction.get(oww_wakeword_key, 0)
            if score >= WAKE_THRESHOLD:
                print(f"\nWake word detected (score {score:.2f})")
                wake_word_detected = True
                reset_wake_word_state()
                break
        except Exception as e:
            print(f"Audio error: {e}")
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def process_with_llm(user_input):
    global conversation_history

    thinking_proc = start_thinking_sound()
    conversation_history.append({"role": "user", "content": user_input})
    if len(conversation_history) > 12:
        conversation_history = conversation_history[-12:]

    response_text = ""
    try:
        stream = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep responses under 40 words. Be direct. No markdown, just plain sentences."
                }
            ] + conversation_history,
            stream=True
        )
        for chunk in stream:
            if "message" in chunk:
                response_text += chunk["message"]["content"]
        conversation_history.append({"role": "assistant", "content": response_text})
    except Exception as e:
        print(f"LLM error: {e}")
        response_text = "I encountered an error processing your request."
    finally:
        stop_thinking_sound(thinking_proc)

    return response_text.strip()


# ---------------------------------------------------------------------------
# Dashboard API (Flask)
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

@app.route("/state")
def api_state():
    with music_lock:
        song   = current_song
        q_len  = len(queue)
        paused = is_paused
        loop   = is_looping
    return jsonify({
        "current_song":              os.path.splitext(os.path.basename(song))[0] if song else None,
        "queue_length":              q_len,
        "is_paused":                 paused,
        "is_looping":                loop,
        "using_bluetooth":           using_bluetooth,
        "volume":                    current_volume,
        "is_listening_for_command":  is_listening_for_command,
    })

@app.route("/volume", methods=["POST"])
def api_volume():
    global current_volume
    data = request.get_json()
    vol  = max(0, min(100, int(data.get("volume", 80))))
    current_volume = vol
    apply_volume(vol)
    return jsonify({"ok": True, "volume": vol})

@app.route("/command", methods=["POST"])
def api_command():
    global wake_word_detected, conversation_history
    data = request.get_json()
    cmd  = data.get("command", "").lower().strip()
    msg  = ""
    if cmd == "pause_toggle":
        if is_paused:
            if resume_music(): msg = "Resuming"
        else:
            if pause_music(): msg = "Paused"
    elif cmd == "skip":     handle_skip();  msg = "Skipping"
    elif cmd == "back":     handle_back();  msg = "Going back"
    elif cmd == "stop":     stop_music();   msg = "Stopped"
    elif cmd == "loop_toggle":
        if is_looping: break_loop(); msg = "Loop off"
        else:
            if loop_song(): msg = "Looping"
    elif cmd == "bluetooth": threading.Thread(target=switch_to_headphones, daemon=True).start(); msg = "Switching to headphones"
    elif cmd == "speaker":   threading.Thread(target=switch_to_speaker,    daemon=True).start(); msg = "Switching to speaker"
    elif cmd == "reset":
        conversation_history = []
        msg = "History cleared"
    elif is_play_command(cmd):
        threading.Thread(target=handle_play_command, args=(cmd,), daemon=True).start()
        song = extract_song_name(cmd)
        msg  = f"Looking for {song}"
    elif is_add_command(cmd):
        threading.Thread(target=handle_add_command, args=(cmd,), daemon=True).start()
        song = extract_song_name(cmd)
        msg  = f"Queuing {song}"
    return jsonify({"ok": True, "message": msg})

@app.route("/listen", methods=["POST"])
def api_listen():
    global wake_word_detected
    wake_word_detected = True
    return jsonify({"ok": True})


@app.route("/recordings")
def api_recordings():
    """Recent command recordings, newest first, each with what was decoded."""
    if not os.path.isdir(COMMAND_RECORDINGS_DIR):
        return jsonify([])
    items = []
    for wav in sorted(os.listdir(COMMAND_RECORDINGS_DIR), reverse=True):
        if not wav.endswith(".wav"):
            continue
        stem = wav[:-len(".wav")]
        entry = {"name": wav, "url": f"/recordings/{wav}"}
        try:
            with open(os.path.join(COMMAND_RECORDINGS_DIR, stem + ".json")) as f:
                entry.update(json.load(f))
        except Exception:
            pass
        items.append(entry)
    return jsonify(items)


@app.route("/recordings/<name>")
def api_recording_file(name):
    # send_from_directory rejects traversal itself; this API listens on
    # 0.0.0.0, so the filename must never be joined onto a path by hand.
    if not name.endswith(".wav"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(COMMAND_RECORDINGS_DIR, name, mimetype="audio/wav")


@app.route("/shutdown", methods=["POST"])
def api_shutdown():
    def do_shutdown():
        time.sleep(1)
        stop_music()
        global script_running, is_listening
        script_running = False
        is_listening   = False
        with open("/tmp/assistant_shutdown", "w") as f:
            f.write("shutdown")
        time.sleep(2)
        sys.exit(0)
    threading.Thread(target=do_shutdown, daemon=True).start()
    return jsonify({"ok": True})

def apply_volume(vol):
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Volume error: {e}")

# ---------------------------------------------------------------------------
# Playlist API routes
# ---------------------------------------------------------------------------

@app.route("/playlists", methods=["GET"])
def api_get_playlists():
    return jsonify(load_playlists())

@app.route("/playlists/create", methods=["POST"])
def api_create_playlist():
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "No name"})
    playlists = load_playlists()
    if name not in playlists:
        playlists[name] = []
        save_playlists(playlists)
    return jsonify({"ok": True})

@app.route("/playlists/add_song", methods=["POST"])
def api_add_to_playlist():
    data     = request.get_json()
    name     = data.get("playlist", "")
    song     = data.get("song", "").strip()
    if not name or not song:
        return jsonify({"ok": False})
    def do_add():
        filepath = resolve_song(song)
        if filepath:
            playlists = load_playlists()
            if name in playlists and filepath not in playlists[name]:
                playlists[name].append(filepath)
                save_playlists(playlists)
    threading.Thread(target=do_add, daemon=True).start()
    return jsonify({"ok": True, "message": f"Adding {song} to {name}"})

@app.route("/playlists/remove_song", methods=["POST"])
def api_remove_from_playlist():
    data  = request.get_json()
    name  = data.get("playlist", "")
    index = int(data.get("index", -1))
    playlists = load_playlists()
    if name in playlists and 0 <= index < len(playlists[name]):
        playlists[name].pop(index)
        save_playlists(playlists)
        return jsonify({"ok": True})
    return jsonify({"ok": False})

@app.route("/playlists/delete", methods=["POST"])
def api_delete_playlist():
    data = request.get_json()
    name = data.get("name", "")
    playlists = load_playlists()
    if name in playlists:
        del playlists[name]
        save_playlists(playlists)
    return jsonify({"ok": True})

@app.route("/playlists/play", methods=["POST"])
def api_play_playlist():
    data = request.get_json()
    name = data.get("name", "")
    playlists = load_playlists()
    if name not in playlists or not playlists[name]:
        return jsonify({"ok": False, "error": "Empty or not found"})
    def do_play():
        stop_music()
        for filepath in playlists[name]:
            if os.path.exists(filepath):
                enqueue_song(filepath)
    threading.Thread(target=do_play, daemon=True).start()
    return jsonify({"ok": True, "message": f"Playing {name}"})

def start_flask():
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=5050, threaded=True, use_reloader=False)

# ---------------------------------------------------------------------------
# Playlist voice commands
# ---------------------------------------------------------------------------

def handle_playlist_command(cmd):
    name      = cmd[len("playlist "):].strip()
    playlists = load_playlists()
    matched   = None
    for pname in playlists:
        if name.lower() == pname.lower():
            matched = pname
            break
    if not matched:
        for pname in playlists:
            if name.lower() in pname.lower() or pname.lower() in name.lower():
                matched = pname
                break
    if matched:
        tracks = [f for f in playlists[matched] if os.path.exists(f)]
        if tracks:
            stop_music()
            for filepath in tracks:
                enqueue_song(filepath)
            speak_with_piper(f"Playing playlist {matched}.")
        else:
            speak_with_piper(f"Playlist {matched} is empty.")
    else:
        speak_with_piper(f"I couldn't find a playlist called {name}.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global is_listening, wake_word_detected, conversation_history, script_running, is_listening_for_command, current_volume

    print("=" * 50)
    print("Voice Assistant")
    print("=" * 50)
    print(f"Wake word : {OWW_MODEL_NAME.replace(chr(95), chr(32))} (openWakeWord)")
    print(f"Mic rate  : {MIC_RATE}Hz -> Vosk: {VOSK_RATE}Hz")
    print(f"Music     : {MUSIC_FOLDER}")
    print("-" * 50)

    # Ensure the project's data/assets folders exist (safe even if the repo
    # already has them checked in).
    os.makedirs(MUSIC_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok=True)
    create_beep_sound()
    set_startup_audio_defaults()
    apply_volume(current_volume)
    initialize_vosk()
    initialize_openwakeword()
    threading.Thread(target=start_flask, daemon=True).start()
    print("Dashboard API running on http://localhost:5050")

    is_listening   = True
    script_running = True

    wake_p, wake_stream, wake_read_size = open_wake_word_stream()
    start_audio_reader(wake_stream)

    try:
        while script_running:
            wake_word_detected = False
            listen_for_wake_word(wake_read_size)

            if not script_running or not is_listening:
                break

            if not wake_word_detected:
                time.sleep(0.1)
                continue

            time.sleep(0.3)
            play_beep()
            is_listening_for_command = True
            command = listen_for_command(timeout_seconds=10)
            is_listening_for_command = False

            if not command:
                print("No command. Back to sleep.")
                continue

            cmd = command.lower().strip()

            # NOTE: voice shutdown/exit is intentionally disabled to prevent
            # accidental shutdowns. Use the dashboard shutdown button instead.

            if cmd in ["start over", "clear history", "forget that", "reset"]:
                conversation_history = []
                speak_with_piper("Okay, starting fresh.")
                continue

            if is_loop_command(cmd):
                handle_loop()
                continue
            if is_break_command(cmd):
                handle_break()
                continue

            if is_pause_command(cmd):
                handle_pause()
                continue
            if is_resume_command(cmd):
                handle_resume()
                continue
            if is_skip_command(cmd):
                handle_skip()
                continue
            if is_back_command(cmd):
                handle_back()
                continue
            if is_stop_music_command(cmd):
                stop_music()
                speak_with_piper("Music stopped.")
                continue

            if is_bt_command(cmd):
                switch_to_headphones()
                continue
            if is_speaker_command(cmd):
                switch_to_speaker()
                continue

            if is_playlist_command(cmd):
                pname = extract_playlist_name(cmd)
                threading.Thread(target=play_playlist, args=(pname,), daemon=True).start()
                continue

            if is_album_command(cmd):
                threading.Thread(target=handle_album_command, args=(command,), daemon=True).start()
                continue
            if is_add_command(cmd):
                threading.Thread(target=handle_add_command, args=(command,), daemon=True).start()
                continue
            if is_play_command(cmd):
                threading.Thread(target=handle_play_command, args=(command,), daemon=True).start()
                continue

            print(f"Unrecognised: '{command}'")
            speak_with_piper("Sorry, I didn't catch that.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        stop_music()
        is_listening   = False
        script_running = False
        stop_audio_reader()
        wake_stream.stop_stream()
        wake_stream.close()
        wake_p.terminate()
        print("Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nShutting down...")
        stop_music()
        sys.exit(0)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
