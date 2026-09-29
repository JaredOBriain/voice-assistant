"""
Benchmark Const-me/Whisper (Direct3D 11, AMD-capable) against faster-whisper.

Runs on the WINDOWS server PC, not the Pi - Const-me is Windows-only and the
whole point is the GPU in this machine. faster-whisper cannot use an RX 590:
CTranslate2 has no native ROCm and the community forks start at gfx900, while
Polaris is gfx803. Const-me sidesteps that with compute shaders.

This only measures. It changes nothing and integrates nothing; see the
decision rule at the bottom of the output for whether integration is even
worth attempting.

Setup on this PC:
  1. Extract cli.zip from Const-me/Whisper release 1.12.0, e.g. C:\\constme\\
  2. Put ggml-medium.en.bin beside it - the GGML equivalent of the medium.en
     that faster-whisper is running, so the comparison is like-for-like
  3. Have transcribe_server.py running locally (it is the thing being raced)
  4. WHISPER_AUTH_TOKEN set, same as the server uses

  python bench_constme.py
  python bench_constme.py --clips 10 --exe C:\\constme\\main.exe
"""
import argparse
import ast
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

import requests

DEFAULTS = {
    "exe":     os.environ.get("CONSTME_EXE",   r"C:\constme\main.exe"),
    "model":   os.environ.get("CONSTME_MODEL", r"C:\constme\ggml-medium.en.bin"),
    "pi":      os.environ.get("PI_URL",        "http://100.73.71.111:5050"),
    "whisper": os.environ.get("WHISPER_URL",   "http://127.0.0.1:5051"),
}

# Clips worth looking at by eye rather than by median. medium.en gets both
# wrong: it returns "All of a sudden." for a spoken "pause", and drops the
# "take" from "play take it easy". A greedy decoder - which Const-me may be,
# since it exposes no beam-size flag - tends to be worse on exactly this kind
# of short utterance, so these decide the accuracy question.
WATCH = {
    "20260918_200122": 'said "pause" - medium.en returns "All of a sudden."',
    "20260918_202458": 'said "play TAKE it easy" - medium.en drops "take"',
}


def server_initial_prompt(path):
    """Pull INITIAL_PROMPT out of transcribe_server.py without importing it.

    Importing would run the module top-level, which constructs WhisperModel
    and loads a 1.4GB model just to read a string. Both engines must get the
    same prompt or the comparison is rigged.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except OSError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "INITIAL_PROMPT":
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        return None
    return None


def list_adapters(exe):
    """Which GPU will it actually use? A silent fall back to the integrated
    adapter would make every number below meaningless."""
    try:
        out = subprocess.run([exe, "-la"], capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()
    except FileNotFoundError:
        sys.exit(f"Const-me CLI not found at {exe}\n"
                 f"Extract cli.zip from the Const-me/Whisper release, or pass --exe")
    except Exception as e:
        return f"(could not list adapters: {e})"


def fetch_clips(pi_url, count, into):
    """Download real captures from the Pi over Tailscale.

    Reuses the assistant's existing /recordings endpoint rather than copying
    files by hand. They are still on the Pi even though SAVE_COMMAND_RECORDINGS
    is now off.
    """
    try:
        listing = requests.get(f"{pi_url}/recordings", timeout=15).json()
    except Exception as e:
        sys.exit(f"Could not reach the Pi at {pi_url}: {e}")
    if not listing:
        sys.exit("The Pi has no saved recordings to benchmark with.")

    clips = []
    for entry in sorted(listing, key=lambda e: e["name"])[:count]:
        name = entry["name"]
        path = os.path.join(into, name)
        urllib.request.urlretrieve(f"{pi_url}/recordings/{name}", path)
        clips.append((name, path, entry.get("seconds", 0)))
    return clips


def run_constme(exe, model, clip, prompt):
    """Time the whole invocation - model load included.

    That is deliberate. Const-me has no daemon mode, so a CLI integration
    would reload the model on every single request, and a number that hides
    that cost would not describe anything we could actually ship.
    """
    cmd = [exe, "-m", model, "-f", clip, "-l", "en", "-nt", "-otxt"]
    if prompt:
        cmd += ["--prompt", prompt]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return time.monotonic() - started, "(timed out)"
    elapsed = time.monotonic() - started

    # -otxt writes beside the input; naming differs between builds, so try
    # both rather than parse stdout, which carries progress noise.
    # utf-8-sig, not utf-8: Const-me writes the .txt with a BOM, and plain
    # utf-8 keeps it as a leading U+FEFF. That is not just untidy - it makes
    # every comparison against faster-whisper's text fail, so identical
    # transcriptions would be reported as disagreements.
    for candidate in (clip + ".txt", os.path.splitext(clip)[0] + ".txt"):
        if os.path.exists(candidate):
            try:
                text = open(candidate, encoding="utf-8-sig", errors="replace").read()
            except OSError:
                continue
            return elapsed, " ".join(text.split()).lstrip("\ufeff")

    text = " ".join(proc.stdout.split()).lstrip("\ufeff")
    if not text and proc.returncode != 0:
        text = f"(failed rc={proc.returncode}: {proc.stderr.strip()[:120]})"
    return elapsed, text


def run_faster_whisper(url, token, clip):
    started = time.monotonic()
    try:
        r = requests.post(f"{url}/transcribe", data=open(clip, "rb").read(),
                          headers={"X-Auth-Token": token}, timeout=(5, 180))
        return time.monotonic() - started, r.json().get("text", "")
    except Exception as e:
        return time.monotonic() - started, f"(failed: {type(e).__name__})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", default=DEFAULTS["exe"])
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--pi", default=DEFAULTS["pi"])
    ap.add_argument("--whisper", default=DEFAULTS["whisper"])
    ap.add_argument("--clips", type=int, default=6)
    args = ap.parse_args()

    token = os.environ.get("WHISPER_AUTH_TOKEN", "")
    if not token:
        print("WARNING: WHISPER_AUTH_TOKEN unset - the server will reject "
              "requests unless it is also running without a token.\n")

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}\n"
                 f"Download ggml-medium.en.bin, or pass --model")

    prompt = server_initial_prompt(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcribe_server.py"))
    print(f"Prompt shared with the server: {'yes' if prompt else 'NO - comparison is unfair'}")

    print("\n=== adapters Const-me can see ===")
    print(list_adapters(args.exe))
    print("Confirm the RX 590 is listed and selected. A silent fall back to "
          "the integrated GPU makes every number below meaningless.\n")

    with tempfile.TemporaryDirectory() as tmp:
        clips = fetch_clips(args.pi, args.clips, tmp)
        print(f"Fetched {len(clips)} clips from the Pi\n")

        # Cold vs warm on the same clip: if the model load dominates, that is
        # the finding, and it rules out the CLI approach rather than Const-me.
        name, path, _ = clips[0]
        cold, _ = run_constme(args.exe, args.model, path, prompt)
        warm, _ = run_constme(args.exe, args.model, path, prompt)
        print(f"Const-me model load overhead: cold {cold:.2f}s, warm {warm:.2f}s "
              f"-> ~{max(cold - warm, 0):.2f}s of it is load\n")

        print(f"{'clip':<20} {'const-me':>9} {'f-whisper':>10}   texts")
        print("-" * 100)
        cm_times, fw_times, disagree = [], [], []
        for name, path, secs in clips:
            cm_t, cm_txt = run_constme(args.exe, args.model, path, prompt)
            fw_t, fw_txt = run_faster_whisper(args.whisper, token, path)
            cm_times.append(cm_t); fw_times.append(fw_t)
            match = cm_txt.strip().lower().rstrip(".") == fw_txt.strip().lower().rstrip(".")
            if not match:
                disagree.append((name, cm_txt, fw_txt))
            print(f"{name[:-4]:<20} {cm_t:8.2f}s {fw_t:9.2f}s   "
                  f"C:{cm_txt[:30]!r}  F:{fw_txt[:30]!r}{'' if match else '  <-- differs'}")

        cm_med, fw_med = statistics.median(cm_times), statistics.median(fw_times)
        print("-" * 100)
        print(f"median: const-me {cm_med:.2f}s vs faster-whisper {fw_med:.2f}s "
              f"-> {fw_med / cm_med:.2f}x {'faster' if cm_med < fw_med else 'SLOWER'}")
        print(f"transcriptions agreeing: {len(clips) - len(disagree)}/{len(clips)}")

        if any(k in n for n, _, _ in disagree for k in WATCH):
            print("\n=== the clips that actually decide accuracy ===")
        for stem, why in WATCH.items():
            for name, cm_txt, fw_txt in disagree:
                if stem in name:
                    print(f"  {why}\n     const-me : {cm_txt[:60]!r}\n     f-whisper: {fw_txt[:60]!r}")

        print("\nDecision rule: worth integrating only if it is BOTH meaningfully "
              "faster than faster-whisper above (load included) AND no worse on "
              "the texts. Faster-but-sloppier is the trade already rejected with "
              "small.en.")


if __name__ == "__main__":
    main()
