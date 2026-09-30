"""
Laptop variant of the transcribe server: faster-whisper only.

The laptop has no Const-me and no GPU worth using, so it runs the same server
with the engine forced. Starting transcribe_server.py there directly would
warn on every boot that Const-me cannot run and fall back on every request -
working, but noisily wrong.

This is a wrapper rather than a second copy on purpose. A copy would have to
receive every future fix twice, and this project has already lost time to the
laptop running a stale version of the server.

    python transcribe_server_laptop.py

Everything else - port, auth token, denoising, the clips page - is whatever
transcribe_server.py does, because it IS transcribe_server.py.
"""
import os

# Must be set before the import: transcribe_server reads WHISPER_ENGINE at
# module level to decide what to announce at startup.
os.environ["WHISPER_ENGINE"] = "faster-whisper"

import transcribe_server as server  # noqa: E402

if __name__ == "__main__":
    server.app.run(host="0.0.0.0", port=server.PORT)
