# Project context for Claude Code

This is a Raspberry Pi 4 voice assistant retrofitted into a 2007 Seat Ibiza,
built incrementally over many sessions. This file captures decisions and
pitfalls that aren't obvious from the code alone.

## Why things are built this way

**Wake word is openWakeWord, not Porcupine.** Picovoice removed their free
tier for personal projects (mid-2026) and now gates signup behind a
company-approval flow. openWakeWord was chosen specifically to avoid any
account/key dependency. Don't suggest Porcupine unless the user asks.

**Command recognition runs two Vosk recognizers in parallel**, not one.
`recognizer` uses a restricted grammar (`VOSK_GRAMMAR`) covering only known
command phrases + `[unk]`, which is far more noise-robust for words like
"skip" or "pause" in a car. `recognizer_full` has no grammar restriction, so
it can transcribe arbitrary song/album/playlist names. `choose_command_text()`
picks which one to trust per-command: name-bearing commands (play, add,
album, playlist) use the full transcription; everything else uses the
restricted one. This dual-recognizer setup was arrived at after both a
single restricted-grammar Vosk (couldn't hear song names — `[unk]` swallowed
them) and a Whisper-based rewrite (accurate on long speech, but bad on short
isolated commands, plus 25+ seconds to transcribe even a 1.7s clip on this
Pi 4) were tried and reverted. **Do not re-suggest Whisper for command
recognition on this hardware** — it was tested (`tiny.en` and `base.en`)
and lost to Vosk on both speed and accuracy for short commands.

**Voice "stop"/"exit"/"quit"/"goodbye" no longer shut down the assistant.**
It used to, and "stop" (meaning "stop the music") kept accidentally killing
the whole process. Voice shutdown is permanently disabled — shutdown only
happens via the dashboard's confirm-guarded button, which writes a flag file
consumed by `assistant-shutdown-watch.service` (running as root, since the
main process doesn't have passwordless poweroff rights).

**The dashboard has three independent on-screen keyboards**, not one keyboard
with mode-switching. An early version tried to reuse a single keyboard
overlay and swap its buttons via JS, which was fragile and kept breaking
(wrong keyboard showing, buttons not updating). Each keyboard
(`osk-overlay` for search, `osk-add-overlay` for adding a song to a playlist,
`osk-playlist-overlay` for naming a new playlist) has its own HTML, its own
show/hide, and its own action button. They only share `oskKey()` /
`oskBackspace()` since only one is ever open at a time. If asked to add a
new keyboard-driven flow, follow this pattern — don't try to make an
existing keyboard multi-purpose again.

**The dashboard is a published HTML file, not a Claude Code artifact
concept.** It's opened directly by Chromium via a `file://` URL in kiosk
mode. It talks to the Flask API on `localhost:5050`. There's no build step.

## Known-fragile areas

- `is_silent()` threshold (50) and `SILENCE_END` timing in
  `listen_for_command()` are tuned by ear for the current Bluetooth headset
  mic. Changing mics will likely require re-tuning.
- yt-dlp breaks silently when YouTube changes something server-side. If
  songs stop being found with no obvious error, `pip install --upgrade
  yt-dlp --break-system-packages` first, before assuming it's a code bug.
- PipeWire source/sink names (`TRUST_MIC_SOURCE`, `BT_HEADPHONE_SOURCE`,
  `BT_SPEAKER_SINK`, `BT_HEADPHONE_CARD`) are hardware-specific strings tied
  to this exact MAC address / USB device. They will not transfer to
  different hardware — get current names with `pactl list sources short` /
  `pactl list sinks short`.
- The Bluetooth headphones button is a **mic toggle only**, not an audio
  output switch — this was a deliberate change from an earlier version that
  switched both mic and speaker output together.

## Style preferences carried over from earlier sessions

- Prefer surgical, targeted edits over rewriting whole files.
- Explain errors clearly when something breaks.
