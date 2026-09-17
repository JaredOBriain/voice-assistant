import ollama
import subprocess
import os
import time
import signal
import json
import wave
import pyaudio
import threading
import sys
import numpy as np
from collections import deque
from flask import Flask, jsonify, request
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

# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
WAKE_WORDS = ["assistant"]

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
MAX_QUEUE        = 30

# ---------------------------------------------------------------------------
# Bluetooth
# ---------------------------------------------------------------------------
BT_SPEAKER_SINK = "bluez_output.78_66_F3_2B_B7_E2.1"

BT_HEADPHONE_MAC  = "A8:F5:E1:6A:ED:64"
BT_HEADPHONE_CARD = "bluez_card.A8_F5_E1_6A_ED_64"
BT_HEADPHONE_SINK = "bluez_output.A8_F5_E1_6A_ED_64.1"

TRUST_MIC_SOURCE    = "alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono"
BT_HEADPHONE_SOURCE = "bluez_input.A8_F5_E1_6A_ED_64.0"

using_bluetooth = False

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
music_lock = threading.Lock()

vosk_model           = None
recognizer           = None
recognizer_full      = None
oww_model            = None
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
current_volume           = 80


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


def choose_command_text(restricted, full):
    """Pick the best transcription. The restricted recognizer reliably detects
    the command word even in noise. If that command carries a free-text name
    (play/add/album/playlist), use the full-vocabulary transcription so the
    name isn't swallowed as [unk]."""
    restricted = (restricted or "").strip()
    full       = (full or "").strip()
    if not restricted:
        return full
    name_bearing = (
        "play album", "add album", "album",
        "add to queue", "add",
        "playlist",
        "play me", "play", "put on",
        "i want to hear", "i want to listen to",
    )
    for prefix in name_bearing:
        if restricted.startswith(prefix):
            return full if full else restricted
    return restricted


def initialize_openwakeword():
    global oww_model
    from openwakeword.model import Model
    import openwakeword
    # One-time model download (safe to call repeatedly; no-op once cached)
    try:
        openwakeword.utils.download_models()
    except Exception as e:
        print(f"Model download note: {e}")
    try:
        oww_model = Model(wakeword_models=[OWW_MODEL_NAME], inference_framework="onnx")
        print(f"openWakeWord loaded. Wake word: '{OWW_MODEL_NAME.replace('_', ' ')}'")
    except Exception as e:
        print(f"openWakeWord init failed: {e}")
        sys.exit(1)


def resample(data):
    if RESAMPLE_FACTOR == 1:
        return data
    audio = np.frombuffer(data, dtype=np.int16)
    audio = np.clip(audio * 2, -32768, 32767).astype(np.int16)
    return audio[::RESAMPLE_FACTOR].tobytes()


def is_silent(data, threshold=50):
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


def check_wake_word(text):
    return any(w in text.lower().strip() for w in WAKE_WORDS)


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

def load_playlists():
    if not os.path.exists(PLAYLIST_FILE):
        return {}
    try:
        with open(PLAYLIST_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Playlist load error: {e}")
        return {}


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

def find_song_in_library(song_name):
    os.makedirs(MUSIC_FOLDER, exist_ok=True)
    needle = song_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    for filename in os.listdir(MUSIC_FOLDER):
        if not filename.lower().endswith(".mp3"):
            continue
        hay = os.path.splitext(filename)[0].lower().replace(" ", "").replace("-", "").replace("_", "")
        if needle in hay or hay in needle:
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
    needle = album_name.lower().replace(" ", "").replace("-", "").replace("_", "")
    matches = []
    for filename in os.listdir(MUSIC_FOLDER):
        if not filename.lower().endswith(".mp3"):
            continue
        hay = filename.lower().replace(" ", "").replace("-", "").replace("_", "")
        if needle in hay:
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

def set_startup_audio_defaults():
    print("Setting startup audio defaults...")
    try:
        subprocess.run(["pactl", "set-default-sink",   BT_SPEAKER_SINK],  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pactl", "set-default-source", TRUST_MIC_SOURCE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Sink:   {BT_SPEAKER_SINK}")
        print(f"Source: {TRUST_MIC_SOURCE}")
    except Exception as e:
        print(f"Startup audio error: {e}")


def is_headphone_connected():
    try:
        result = subprocess.run(
            ["bluetoothctl", "info", BT_HEADPHONE_MAC],
            capture_output=True, text=True, timeout=5
        )
        return "Connected: yes" in result.stdout
    except Exception as e:
        print(f"BT check error: {e}")
        return False


def switch_to_headphones():
    global using_bluetooth

    if using_bluetooth:
        switch_to_speaker()
        return

    if not is_headphone_connected():
        speak_with_piper("Bluetooth headphones are not connected.")
        return

    try:
        subprocess.run(
            ["pactl", "set-card-profile", BT_HEADPHONE_CARD, "headset-head-unit"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)

        subprocess.run(
            ["pactl", "set-default-source", BT_HEADPHONE_SOURCE],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        using_bluetooth = True
        print("Switched mic to Bluetooth headset.")
        speak_with_piper("Switched to headset microphone.")

    except subprocess.CalledProcessError as e:
        print(f"BT mic switch error: {e}")
        speak_with_piper("Failed to switch to headset microphone.")


def switch_to_speaker():
    global using_bluetooth

    try:
        subprocess.run(
            ["pactl", "set-default-source", TRUST_MIC_SOURCE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["pactl", "set-card-profile", BT_HEADPHONE_CARD, "a2dp-sink"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        using_bluetooth = False
        print("Reverted to USB microphone.")
        speak_with_piper("Switched back to USB microphone.")

    except Exception as e:
        print(f"Mic revert error: {e}")


# ---------------------------------------------------------------------------
# Voice recognition
# ---------------------------------------------------------------------------

def listen_for_command(timeout_seconds=10):
    p      = pyaudio.PyAudio()
    stream = open_mic_stream(p)
    stream.start_stream()

    print("\nListening for command...")

    start_time      = time.time()
    silence_start   = None
    speech_detected = False
    final_text      = ""

    recognizer.Reset()
    recognizer_full.Reset()

    def finalize():
        r = json.loads(recognizer.FinalResult()).get("text", "")
        f = json.loads(recognizer_full.FinalResult()).get("text", "")
        return choose_command_text(r, f)

    while is_listening:
        if time.time() - start_time > timeout_seconds:
            print("Timeout.")
            final_text = finalize()
            break

        raw = stream.read(8000, exception_on_overflow=False)

        if is_silent(raw):
            if speech_detected and silence_start is None:
                silence_start = time.time()
            elif silence_start and time.time() - silence_start > 1.5:
                final_text = finalize()
                if final_text:
                    print(f"Command: {final_text}")
                break
            continue

        data = resample(raw)

        done_r = recognizer.AcceptWaveform(data)
        done_f = recognizer_full.AcceptWaveform(data)

        if done_r or done_f:
            final_text = finalize()
            if final_text:
                speech_detected = True
                print(f"Command: {final_text}")
                break
        else:
            partial = json.loads(recognizer_full.PartialResult()).get("partial", "")
            if partial:
                silence_start = None
                speech_detected = True
                print(f"Hearing: {partial}    ", end='\r')
            elif speech_detected and silence_start is None:
                silence_start = time.time()
            elif silence_start and time.time() - silence_start > 1.5:
                final_text = finalize()
                if final_text:
                    print(f"Command: {final_text}")
                break

    stream.stop_stream()
    stream.close()
    p.terminate()
    return final_text.strip()


def continuous_listen_for_wake_word():
    global wake_word_detected

    CHUNK_16K = 1280
    read_size = CHUNK_16K * RESAMPLE_FACTOR

    p      = pyaudio.PyAudio()
    stream = open_mic_stream(p, frames_per_buffer=read_size)
    stream.start_stream()

    print(f"\nListening for wake word: '{OWW_MODEL_NAME.replace('_', ' ')}'")
    oww_model.reset()

    while is_listening:
        if wake_word_detected:
            break
        try:
            raw = stream.read(read_size, exception_on_overflow=False)
            pcm = np.frombuffer(raw, dtype=np.int16)[::RESAMPLE_FACTOR]
            prediction = oww_model.predict(pcm)
            score = prediction.get(OWW_MODEL_NAME, 0)
            if score >= WAKE_THRESHOLD:
                print(f"\nWake word detected (score {score:.2f})")
                play_beep()
                wake_word_detected = True
                oww_model.reset()
                break
        except Exception as e:
            print(f"Audio error: {e}")
            time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    p.terminate()


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
    initialize_vosk()
    initialize_openwakeword()
    threading.Thread(target=start_flask, daemon=True).start()
    print("Dashboard API running on http://localhost:5050")

    is_listening   = True
    script_running = True

    try:
        while script_running:
            wake_word_detected = False
            continuous_listen_for_wake_word()

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

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        stop_music()
        is_listening   = False
        script_running = False
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
