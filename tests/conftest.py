"""Shared pytest configuration.

``asyncio_mode = auto`` means async tests need no ``@pytest.mark.asyncio``
decorator, though one is included on the async tests here for clarity.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Solid colours, one per second of clip. `video_frames` samples by seeking, so a
# clip built from these makes an evenly spaced sample self-evident: the colour of
# each returned frame says which second it came from.
CLIP_COLOURS = ("red", "green", "blue", "yellow", "magenta", "cyan")


@pytest.fixture
def make_clip(tmp_path):
    """Build a real H.264 clip, one solid colour per second.

    Frame extraction cannot be tested with a fake container, but the tests must not
    fail on a machine without ffmpeg either, so this skips instead.

    Every frame is a keyframe (``-g 1 -bf 0``) because ``video_frames`` seeks with
    ``CAP_PROP_POS_FRAMES``, which lands on the nearest keyframe. Without that the
    sampled pixel colours would depend on the encoder's GOP layout.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to build a real clip")

    def build(name: str = "colour.mp4", *, seconds: int = 3, fps: int = 10, size: str = "160x120") -> Path:
        assert 1 <= seconds <= len(CLIP_COLOURS)
        inputs: list[str] = []
        for colour in CLIP_COLOURS[:seconds]:
            inputs += ["-f", "lavfi", "-i", f"color=c={colour}:s={size}:d=1:r={fps}"]
        graph = "".join(f"[{i}:v]" for i in range(seconds)) + f"concat=n={seconds}:v=1:a=0[out]"

        path = tmp_path / name
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
             "-filter_complex", graph, "-map", "[out]",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-g", "1", "-bf", "0", str(path)],
            check=True,
            capture_output=True,
        )
        return path

    return build


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: async test")
