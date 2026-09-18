#!/usr/bin/env python3
"""Bulk-import a YouTube / YouTube Music playlist into the assistant.

    python3 import_playlist.py <playlist-url> [playlist-name]

Downloads every track into data/Music and writes the result into
data/playlists.json, so it appears on the dashboard and answers to
"hey jarvis, playlist <name>".

Pass a short playlist-name if you plan to ask for it by voice — YouTube
playlist titles tend to be long, and the voice matcher has to hear the
whole thing. Re-running on the same URL is safe: already-downloaded tracks
are skipped and the playlist is topped up rather than replaced.
"""

import os
import subprocess
import sys
import tempfile

from assistant import (
    MUSIC_FOLDER,
    YTDLP_JS_RUNTIME,
    YTDLP_PATH,
    find_song_in_library,
    load_playlists,
    save_playlists,
)

MAX_TRACK_SECONDS = 900  # skip hour-long DJ sets / full-album uploads


def probe_playlist(url):
    """Playlist title and track titles, without downloading anything."""
    result = subprocess.run(
        [YTDLP_PATH, *YTDLP_JS_RUNTIME, "--flat-playlist",
         "--print", "%(playlist_title)s\t%(title)s", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Could not read that playlist:\n{result.stderr.strip()}")
    rows = [line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line]
    if not rows:
        sys.exit("No playlist entries found at that URL — is it a single video?")
    # yt-dlp reports "NA" for entries it can't see (deleted/private). Dropping
    # them isn't just tidiness: "NA" reduces to the single word "na", which
    # find_song_in_library() will happily match against any library filename
    # containing that word, silently importing an unrelated track.
    titles = [title.strip() for _, title in rows if title.strip() != "NA"]
    unavailable = len(rows) - len(titles)
    if unavailable:
        print(f"Skipping {unavailable} unavailable (deleted or private) entries.")
    return rows[0][0].strip(), titles


def download_playlist(url, path_file):
    """Download the whole playlist as mp3s, with progress left on screen.

    yt-dlp writes each finished file's path to path_file: 'after_move' is the
    stage after the mp3 conversion, so the paths are the final .mp3 names
    rather than the .webm/.m4a ones it downloaded first."""
    subprocess.run([
        YTDLP_PATH, url,
        *YTDLP_JS_RUNTIME,
        "--yes-playlist",
        "--ignore-errors",  # one private/deleted video shouldn't sink the import
        "--extract-audio", "--audio-format", "mp3", "--audio-quality", "0",
        "--output", os.path.join(MUSIC_FOLDER, "%(title)s.%(ext)s"),
        "--match-filter", f"duration < {MAX_TRACK_SECONDS}",
        "--no-simulate", "--print-to-file", "after_move:filepath", path_file,
    ])
    if not os.path.exists(path_file):
        return []
    with open(path_file) as f:
        return [line.strip() for line in f if line.strip()]


def collect_tracks(titles, downloaded):
    """Playlist-ordered list of files on disk.

    Resolving each title against the library (rather than trusting the
    download list alone) keeps the playlist in its original order and picks
    up tracks that were already present from an earlier run, which yt-dlp
    skips silently."""
    tracks = []
    for title in titles:
        found = find_song_in_library(title)
        if found and found not in tracks:
            tracks.append(found)
    # Catch anything whose filename diverged too far from its title to match.
    for path in downloaded:
        if path not in tracks and os.path.exists(path):
            tracks.append(path)
    return tracks


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    url = sys.argv[1]
    playlist_title, titles = probe_playlist(url)
    name = sys.argv[2].strip() if len(sys.argv) > 2 else playlist_title
    if not name or name == "NA":
        sys.exit("Couldn't read a title for that playlist — pass a name as the second argument.")

    print(f"\n'{playlist_title}' — {len(titles)} tracks — importing as playlist '{name}'\n")
    os.makedirs(MUSIC_FOLDER, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = download_playlist(url, os.path.join(tmp, "paths.txt"))

    tracks = collect_tracks(titles, downloaded)
    if not tracks:
        sys.exit("\nNothing was downloaded — playlist left unchanged.")

    playlists = load_playlists()
    existing  = playlists.get(name, [])
    playlists[name] = existing + [t for t in tracks if t not in existing]
    save_playlists(playlists)

    added   = len(playlists[name]) - len(existing)
    missing = len(titles) - len(tracks)
    print(f"\nPlaylist '{name}': {added} new tracks ({len(playlists[name])} total).")
    if missing > 0:
        print(f"{missing} of {len(titles)} tracks could not be downloaded "
              f"(private, region-locked, or longer than {MAX_TRACK_SECONDS // 60} minutes).")


if __name__ == "__main__":
    main()
