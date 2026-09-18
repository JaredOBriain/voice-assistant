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
and lost to Vosk on both speed and accuracy for short commands. Whisper
*does* now transcribe song/album names, but only by being offloaded to the
home PC (see below); nothing Whisper-shaped runs on the Pi itself.

**Free-text names are offloaded to a Whisper server on the home PC.**
`server/transcribe_server.py` runs faster-whisper on the home PC and is
reached over Tailscale at `WHISPER_SERVER_URL`. Only the name-bearing half
of a command goes over the wire — `finalize()` in `listen_for_command()`
calls it only when `is_name_bearing()` matched, so control words like
"skip"/"pause" never leave the Pi and still work with the car parked out of
signal. Local Vosk is the automatic fallback whenever the server is
unreachable or too slow, so a dead link degrades quality rather than
breaking commands. The shared secret is `WHISPER_AUTH_TOKEN`, kept in a
gitignored `.env` that `assistant.service` loads via `EnvironmentFile` — it
is deliberately not hardcoded anywhere, because **this repo is public**.
Don't reintroduce it as a default value in `assistant.py`. On a fresh
checkout, create `.env` with `WHISPER_AUTH_TOKEN=<token>`; without it the
Pi simply falls back to local Vosk for names.

**Whisper output is punctuated; the command matchers are not.**
faster-whisper returns things like `"Play, Bohemian Rhapsody."`, and every
matcher downstream (`is_play_command`, `extract_song_name`, the `_words()`
filename matching) works on bare lowercase words. A single inserted comma
is enough to stop `"play, x"` matching the `"play "` prefix, sending a
perfectly good command to "Sorry, I didn't catch that". `strip_punctuation()`
in `transcribe_remote()` exists for exactly this — don't remove it, and note
it deliberately keeps apostrophes, since titles need them ("Livin' On A
Prayer").

**There is exactly one mic stream, opened once for the life of the process.**
`open_wake_word_stream()` is called once in `main()`, and both
`listen_for_wake_word()` and `listen_for_command()` read from that same
stream. Two reasons it must stay that way: this mic cannot be opened twice
at once (a second PortAudio stream while the first is held open fails with
"Device unavailable" and crashes the process), and rebuilding the stream
every wake cycle produced a startup artifact that openWakeWord misread as
the wake word. The cost of one persistent stream is that audio keeps
buffering while the assistant is off handling a command or speaking, so both
functions drain `stream.get_read_available()` before they start — without
that, stale backlog gets processed as if it had just been spoken.

**`oww_model.reset()` does not reset what its name suggests.** In
openwakeword 0.4.0 it clears only `prediction_buffer`, leaving
`Model.preprocessor`'s `raw_data_buffer`, `melspectrogram_buffer` and
`feature_buffer` intact — roughly 10 seconds of audio and embedding history.
That meant the embeddings of the utterance that had just triggered a
detection were still inside the classifier's window when the next listening
session began, re-firing a second detection within its first few chunks.
This was the real cause of "hey jarvis" triggering twice, and it survived an
earlier fix that was wrongly assumed to have solved it. Always reset via
`reset_wake_word_state()`, which clears the preprocessor buffers too; don't
"simplify" it back to a bare `oww_model.reset()`.

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

- `SILENCE_THRESHOLD` (currently 400) and the 1.5s end-of-speech timing in
  `listen_for_command()` are tuned by ear for the current mic. Changing mics
  will likely require re-tuning. Note `is_silent()` only decides *when a
  command has ended* — every chunk is fed to both recognizers regardless,
  because ambient cabin noise sits too close to speech level for amplitude
  alone to safely gate what Vosk hears.
- Playlists store absolute paths, but `load_playlists()` rebuilds the
  directory part from the current `MUSIC_FOLDER` on every read, keeping the
  filename as a track's real identity. This is not cosmetic: moving the
  project once left every entry pointing at the old `~/Music` location, and
  since every play path guards with `os.path.exists()`, all ten playlists
  silently reported themselves empty rather than erroring. The rewrite means
  the next `save_playlists()` quietly migrates the file, and it only holds
  because tracks live directly in `MUSIC_FOLDER` with no subdirectories.
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

## Adding music in bulk

`import_playlist.py` already exists for this — don't rebuild it, and don't
reach for the Spotify API or `spotdl`:

    python3 import_playlist.py <youtube-playlist-url> [playlist-name]

It downloads a whole YouTube/YouTube Music playlist into `data/Music` and
writes the entry into `playlists.json`, reusing `assistant.py`'s own
`MUSIC_FOLDER`, `find_song_in_library()` and `save_playlists()` rather than
duplicating that logic (importing `assistant` is side-effect free — it opens
no audio devices and starts no threads). Re-running is safe: yt-dlp skips
tracks already downloaded and the playlist is topped up rather than replaced.

Two non-obvious bits. Tracks are matched back to files by title *after* the
download, so an interrupted run resumes correctly instead of losing the
tracks it already had. And entries YouTube reports as `NA` (deleted or
private videos) are filtered out early — `"NA"` reduces to the single word
`"na"`, which `find_song_in_library()` will happily match against any
filename containing that word, silently importing an unrelated track.

Pass a short playlist name if it's going to be used by voice; YouTube
playlist titles are long, and the voice matcher has to hear the whole thing.

## Style preferences carried over from earlier sessions

- Prefer surgical, targeted edits over rewriting whole files.
- Explain errors clearly when something breaks.
