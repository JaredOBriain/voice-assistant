# Voice Assistant

A Raspberry Pi voice assistant for a car head unit: wake-word activated,
plays/downloads music by voice, dashboard touchscreen UI, playlists, and
Bluetooth mic switching.

## Hardware

- Raspberry Pi 4
- WaveShare touchscreen (HDMI, labwc/Wayland desktop)
- USB microphone (C-Media Trust Gaming Microphone) + Bluetooth headset mic
- Car head unit audio via 3.5mm / high-low adapter

## Stack

- **Wake word:** [openWakeWord](https://github.com/dscripka/openWakeWord) (`hey_jarvis`, fully offline, no account)
- **Command recognition:** [Vosk](https://alphacephei.com/vosk/) — a restricted-grammar recognizer for noise-robust
  command words, running alongside a full-vocabulary recognizer for song/playlist names.
  See `choose_command_text()` in `assistant.py` for why two recognizers are used.
- **TTS:** [Piper](https://github.com/rhasspy/piper)
- **Playback:** `mpg123`
- **Song download:** `yt-dlp` — **keep this updated**, YouTube changes break old versions
  silently (see Known Issues below)
- **Dashboard:** single-file HTML/JS app, served as a local file, talking to a Flask API
  on port 5050

## Project layout

```
assistant.py              Main process: wake word, voice commands, music engine, Flask API
dashboard/dashboard.html  Touchscreen dashboard (opened directly in Chromium, kiosk mode)
systemd/                  Service files — copy to /etc/systemd/system/
requirements.txt          Python dependencies
piper/                    Piper binary + voice model (not in git — see Setup)
models/                   Vosk model folder (not in git — see Setup)
assets/                   beep.wav (auto-generated), process.wav (thinking sound, add manually)
data/                     Runtime data — Music/, playlists.json (not in git, user-specific)
```

## Setup

### 1. System packages

```bash
sudo apt update
sudo apt install python3-pip mpg123 build-essential cmake
```

### 2. Python dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Fetch the models/binaries not tracked in git

- **Piper**: download a release from https://github.com/rhasspy/piper and the
  `en_US-lessac-medium` voice files, place them in `piper/`
- **Vosk model**: download `vosk-model-small-en-us-0.15` from
  https://alphacephei.com/vosk/models, extract into `models/`
- **openWakeWord models**: downloaded automatically on first run
- **yt-dlp**: install per https://github.com/yt-dlp/yt-dlp, and keep it updated:
  ```bash
  pip install --upgrade yt-dlp --break-system-packages
  ```

### 4. Audio config

Edit the constants near the top of `assistant.py` to match your hardware:
`MIC_DEVICE_INDEX`, `TRUST_MIC_SOURCE`, `BT_HEADPHONE_*`, `BT_SPEAKER_SINK`.
Find current PipeWire names with:
```bash
pactl list sources short
pactl list sinks short
```

### 5. Passwordless shutdown

The dashboard's shutdown button needs `systemctl poweroff` to work without a
password prompt — this is usually fine by default on Raspberry Pi OS. Test with:
```bash
systemctl poweroff
```

### 6. Install the services

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now assistant.service
sudo systemctl enable --now assistant-shutdown-watch.service
```

### 7. Kiosk dashboard on boot

Add to `~/.config/labwc/autostart`:
```bash
sleep 5 && chromium --kiosk --noerrdialogs --disable-infobars --password-store=basic file:///home/jpie/voice-assistant/dashboard/dashboard.html &
```

## Running / debugging

```bash
journalctl -u assistant.service -f
```

To leave kiosk mode for debugging on the touchscreen: **Alt+F4**.

## Known issues / gotchas

- **yt-dlp goes stale.** If songs stop being found with no visible error,
  update yt-dlp first — this has been the cause every time so far.
- **Voice shutdown is intentionally disabled** (saying "stop" used to kill the
  whole assistant by accident). Only the dashboard button can power off the Pi.
- **Silence threshold (`is_silent`, currently 50)** is tuned for a close
  Bluetooth headset mic. If command recognition seems to cut off early or
  never stop, this is the first thing to adjust.
