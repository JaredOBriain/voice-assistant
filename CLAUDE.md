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

**Noise suppression lives on the server, not the Pi.** The Pi runs Vosk in
real time and has no CPU to spare, and its restricted-grammar recogniser is
already the noise-robust path — the weak link is free-text names, which go to
Whisper anyway. So `transcribe_server.py` denoises (spectral gating via
`noisereduce`) before transcribing, costing the Pi nothing. `DENOISE_STRENGTH`
is deliberately below 1.0: gating hard enough to erase the noise also carves
holes in the speech, and Whisper reads those artefacts as words.

To hear the difference, open **`http://<pc>:5051/`** in a browser on the home
PC: the server keeps the raw and denoised version of every clip it receives
(in `server/clips/`, newest 30, gitignored) and lists them side by side with
players. Only name-bearing commands ever reach the server, so "pause" and
"skip" will never appear there. `/denoise` and `?denoise=0` also exist for
scripted A/B, but both are POST-only and need the auth header — they cannot
be opened in a browser.

If `noisereduce` isn't installed the server transcribes raw audio and says so
at startup — it must never fail the request, because the Pi would then fall
back to its much weaker local Vosk transcription. Note denoising adds to the
server's response time, which is bounded by the Pi's `WHISPER_READ_TIMEOUT`;
the per-request timing log exists to keep that visible.

**`small.en` was tried on the server and rejected — don't re-propose it.**
It measured 3x faster (10.5s to 3.5s) and looked fine on a six-clip bench,
but in real use it needed repeating far too often. The bench was misleading
because it contained almost no long song titles, which is the entire reason
the server exists. Likewise `beam_size=1`: only 7% faster than 5, because
beam search touches the decoder and a spoken command emits a handful of
tokens — the encoder is the cost. Both are back at `medium.en` / `beam_size=5`
and ~11s per request is the accepted price of accuracy.

Clip length is **not** a lever: Whisper always encodes a padded 30-second
window, so 3s and 25s of audio cost the same (measured 10.66s vs 11.88s).
Only model size and hardware change that — and hardware is what finally won.

**The RX 590 does help, via Const-me/Whisper — but not through
faster-whisper.** faster-whisper is built on CTranslate2, which has no native
ROCm, and every community ROCm fork starts at gfx900 while this card is
Polaris/gfx803. Const-me/Whisper sidesteps that entirely by running Whisper on
Direct3D 11 compute shaders, so any DX11 GPU works. Measured **~2x faster than
medium.en in practice at accuracy judged equal or better**, at the same model
size — which is why it costs nothing, unlike `small.en`.

`ENGINE` in `transcribe_server.py` selects the engine and defaults to
`constme`, since the desktop is the preferred server and has the GPU. The
laptop has neither and runs **`transcribe_server_laptop.py`**, a thin wrapper
that forces `faster-whisper` through `WHISPER_ENGINE` — a wrapper rather than a
second copy, so fixes cannot land in one and not the other. faster-whisper
stays loaded either way as the per-request fallback: an experimental engine
must never cost the Pi its transcription, because it would drop to much weaker
local Vosk. `?engine=constme` / `?engine=faster-whisper` overrides for one
request, which is how the two were compared on identical audio.

Const-me needs its own setup, none of it automatic. `CONSTME_MODEL` has **no
default** because which GGML model is used is a real choice and a guessed path
would silently transcribe with the wrong one. It wants **GGML** models, not the
CTranslate2 ones faster-whisper downloads. `cli.zip` from release 1.12.0 ships
prebuilt, so nothing needs compiling, and `server/setup_constme.ps1` fetches
both it and the model. Its `-otxt` output carries a **UTF-8 BOM**: read with
`utf-8-sig` or every text comparison fails for no visible reason. It exposes no
beam-size flag, so it may decode greedily — that did not hurt accuracy in
practice, but it is why the comparison was done by ear rather than assumed.
Last Const-me release is July 2023, so it is unmaintained; the binary is
self-contained, which is why that matters less than it would for a scraper.

One route that genuinely is a dead end: whisper.cpp via Vulkan does support
Polaris, but it measured ~13x slower on Windows than Linux, so it is only worth
revisiting if that PC ever runs Linux.

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

**The Bluetooth headset is pinned to HFP, and that costs playback quality
on purpose.** The Shokz OpenRun gives you *either* 48kHz stereo A2DP
playback *or* its HFP microphone, never both — classic Bluetooth runs one
or the other over the link. Worse, the choice is made when the profile
connects, not when it's switched: once A2DP is connected the mic node is
never created, and switching the card profile to `headset-head-unit`
afterwards does not bring it back. Only a reconnect does. Its mic is much
better than the USB one, so the mic wins; music goes to the JBL car speaker
(`BT_SPEAKER_SINK`), which does its own 48kHz A2DP.

A2DP outranks HFP on profile priority (18 vs 3) and WirePlumber re-picks by
priority on every start (`hooks.device.profile.state` is disabled), so
without help the headset silently connects as a speaker and the assistant
loses its best mic. `~/.config/wireplumber/wireplumber.conf.d/51-shokz-hfp.conf`
pins it to `headset-head-unit` by MAC. That file lives outside the repo —
**on a fresh install it has to be recreated**, or the mic won't be there.
Nothing in `switch_to_headphones()` / `switch_to_speaker()` may call
`set-card-profile` on this headset: flipping it to `a2dp-sink` destroys the
mic node until the next reconnect.

**The headset's PipeWire source name is not stable.** It has appeared as
both `bluez_input.A8_F5_E1_6A_ED_64.0` and `bluez_input.A8:F5:E1:6A:ED:64`
(underscores vs colons) across reboots on this same hardware. A hardcoded
name silently stops matching and the mic switch just fails, so
`find_bt_mic_source()` resolves it by MAC and accepts either spelling.
Don't replace it with a constant.

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

- **`SILENCE_THRESHOLD`, `MIC_GAIN` and `HIGHPASS_HZ` are one coupled set —
  never change one alone.** `is_silent()` compares raw mean-abs amplitude
  against `SILENCE_THRESHOLD` to decide *when a command has ended*, so
  raising the gain without raising the threshold makes silence look like
  speech and no command ever finishes before the 10s timeout. The current
  values are measured, not guessed: after the high-pass and `MIC_GAIN`,
  silence sits at 636-715 mean-abs and speech at 1160+, so 900 splits them.
  Re-derive from real captures in `data/command_recordings/` after any
  change. Note `is_silent()` only gates the end-of-speech timer — every
  chunk reaches both recognizers regardless, because cabin noise sits too
  close to speech level for amplitude alone to decide what Vosk hears.
- The mic feed is high-passed before it is boosted, and the order matters.
  Captured commands measured ~-22 dBFS peak at 7-9 dB SNR, with the noise
  dominated by sub-100Hz rumble — 50Hz mains hum sat ~37 dB above the noise
  median, with harmonics at 100 and 150Hz. That rumble carries no speech but
  set the peak level, which is why speech was so quiet. Boosting first would
  simply have amplified the hum and clipped on it. `prepare_mic_audio()`
  high-passes at 100Hz, then applies `MIC_GAIN`, landing speech near -6 dBFS.
  Its filter state deliberately carries across chunks and is cleared per
  capture by `reset_mic_filter()`; restarting the filter every 8000-sample
  chunk would inject a discontinuity every 167ms.
- Gain lifts speech *and* noise equally — only the high-pass improves SNR
  (measured 8.8 -> 11.1 dB on real speech). Don't expect a louder signal to
  be a cleaner one.
- The wake-word path does **not** use `prepare_mic_audio()`; it reads the raw
  stream. That is deliberate — `WAKE_THRESHOLD` is tuned against unprocessed
  audio, and changing the levels feeding openWakeWord risks reviving the
  false-trigger problems above.
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
- **yt-dlp needs a JavaScript runtime, and the Pi uses quickjs.** Without
  one, yt-dlp can't solve YouTube's signature challenges, quietly falls back
  to clients like `visionos`/`m3u8`, and most downloads die on
  `HTTP Error 403: Forbidden` — intermittently, which makes it look like a
  network problem rather than a missing dependency. Only `deno` is enabled
  by default and it isn't packaged for Debian; `nodejs` is, but trixie ships
  20.x and yt-dlp requires >= 22.0.0 (it logs
  `JS runtimes: node-20.19.2 (unsupported)`). quickjs is in Debian main at
  2025.04.26 against a 2023.12.09 minimum, so: `sudo apt install quickjs`.
  `YTDLP_JS_RUNTIME` in `assistant.py` carries the flag and is shared with
  `import_playlist.py` — every yt-dlp invocation must include it, since the
  runtime is not picked up automatically. Diagnose with
  `yt-dlp -v ... 2>&1 | grep "JS runtimes:"`.
- Never run `apt autoremove` on this Pi. It considers the kernel headers and
  several GNOME menu packages to be orphans and will happily remove them.
  Remove packages by name instead.
- PipeWire names (`TRUST_MIC_SOURCE`, `BT_SPEAKER_SINK`) and
  `BT_HEADPHONE_MAC` are hardware-specific, tied to this exact MAC address /
  USB device. They will not transfer to different hardware — get current
  names with `pactl list sources short` / `pactl list sinks short`. The
  headset's *source* is deliberately not a constant; see
  `find_bt_mic_source()` above.
- Adding a second Bluetooth audio device *mid-session* often fails, while
  both connect fine from a clean boot. Connecting the JBL while the headset
  already held its HFP link failed every time with bluetoothd logging
  `a2dp_select_capabilities() Unable to select SEP`, and since the JBL
  advertises only A2DP it then dropped the connection entirely. The headset
  hit the same error on its first connect. **Reboot with both devices
  powered on** rather than debugging it — SCO (headset mic) and A2DP (JBL
  music) genuinely do run at once once negotiated from scratch.
- The Bluetooth headphones button is a **mic toggle only**, not an audio
  output switch — this was a deliberate change from an earlier version that
  switched both mic and speaker output together. It now only moves the
  default source between the headset and USB mics; it must never touch the
  headset's card profile.
- `set_startup_audio_defaults()` prefers the headset mic when its source
  exists and falls back to the USB mic when it doesn't, so the assistant is
  never left deaf if the headset is off. That fallback is the only thing
  keeping voice control alive when the headset's battery dies.

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

## Diagnosing misheard commands

Every command capture is saved to `data/command_recordings/` as a wav plus a
json sidecar, and served by the Flask API:

    curl http://localhost:5050/recordings          # newest first, with decodes
    curl -O http://localhost:5050/recordings/<name>.wav

The wav is the same 16kHz mono PCM `transcribe_remote()` posts to the Whisper
server, so it is exactly what the server heard — not a re-encode. The sidecar
records what each recogniser made of that audio (`restricted` from the
grammar-limited Vosk, `full` from the free one, `remote` from Whisper or null
when it wasn't consulted, and the `final` chosen text), which is what makes a
misheard command diagnosable: you can tell a bad recording apart from a good
recording that was decoded badly.

Captures that decoded to nothing are kept too — "it didn't hear me" is the
case most worth listening back to. Only the most recent
`MAX_COMMAND_RECORDINGS` (30) are kept, so the SD card can't fill; at ~10s
worst case per clip that caps out around 10MB. `data/` is gitignored, so none
of this is committed.

Note `/recordings/<name>` uses `send_from_directory` deliberately: the API
listens on `0.0.0.0`, so a hand-joined path would be a traversal hole.

## Podcasts

Say *"podcast \<search terms\>"*; it searches YouTube, reads back the title and
length, and only downloads once you answer **yes**. *"resume podcast"* (or
"continue podcast") returns to the most recent unfinished episode. Re-requesting
by name starts fresh — resuming is deliberately a separate command.

**Podcasts have their own player, and that is not duplication.** The music
engine runs `mpg123 -q` fire-and-forget with output discarded, so nothing knows
how far into a track it is, and pause is SIGSTOP — the position lives inside a
frozen process and dies with it. An hour-long episode spans several drives, so
it needs `mpg123 -R` (remote mode): commands on stdin (`LOAD`, `JUMP`, `PAUSE`,
`QUIT`) and position on stdout as `@F <frame> <frames-left> <secs> <secs-left>`.
Positions are stored as **frames**, because `JUMP` takes frames.

Three things about remote mode that are not obvious and were each found the
hard way:

- **Frames-left counts down to 1, never 0.** Testing `frames_left <= 0` to
  detect the end silently never fires. End-of-track is decided by *position*
  (`duration - seconds <= 1.0`) instead.
- **`@P 0` means stopped, but so does the end of a track.** `@P 1` is paused and
  `@P 2` is playing, so pausing is safe to ignore — but distinguishing "finished"
  from "stopped early" still needs the position check above, and it decides
  whether the file gets deleted.
- **mpg123 does not exit when a track ends** — it idles waiting for the next
  command. It must be explicitly retired, or `podcast_is_active()` keeps
  claiming the audio device and pause/stop route to a finished episode.

**Members-only uploads are filtered on `availability`, not on titles.**
yt-dlp reports `availability=subscriber_only` in `--flat-playlist` output, so
`UNPLAYABLE_AVAILABILITY` gates on the field rather than pattern-matching
"MEMBERS" or "AD FREE" in the title. Podcast channels post a lot of these near
the top of their feed — one real listing was 10 unplayable out of 18 — which is
why the channel listing deliberately fetches several times more rows than it
intends to offer.

**Searching uses `--flat-playlist`, not a full extraction.** Extracting each
result to read its title took over 90 seconds on this Pi and timed out; reading
the results page takes about 4. YouTube mixes *channels* into search results —
they come back with a `/channel/` URL and `NA` duration and cannot be
downloaded, so they are filtered out rather than offered.

**Do not reuse `download_from_youtube()` for episodes.** Its
`--match-filter "duration < 600"` rejects anything over ten minutes, which is
every podcast. `download_podcast()` uses `MAX_PODCAST_SECONDS` and encodes mono
at a speech bitrate — roughly 30MB an hour, measured, against the songs path's
full-quality encode.

**Episodes live in `data/Podcasts/`, never `MUSIC_FOLDER`.** This is not
tidiness: `find_song_in_library()` word-matches every file in the music folder,
so an episode sitting there could be returned for an ordinary song request.

**Predicate ordering is load-bearing.** `is_podcast_resume_command()` must be
tested before `is_resume_command()`, which matches a bare "resume" and would
otherwise resume the music. Both podcast predicates are also matched *before*
the loop/pause/skip/back matchers, because those use substring tests and a
spoken episode title can easily contain "back" or "stop". The ordering holds in
two places that must stay in step: the voice chain in `main()` and the
`/command` hub the dashboard shares.

Finished episodes delete themselves; stopping part-way keeps both the file and
the position. `data/podcasts.json` is keyed by video URL and mirrors the
playlists pattern — read fresh, whole-file write, never raises.

## Style preferences carried over from earlier sessions

- Prefer surgical, targeted edits over rewriting whole files.
- Explain errors clearly when something breaks.
