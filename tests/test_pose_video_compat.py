from __future__ import annotations

import subprocess
from pathlib import Path

from src import pose


def test_make_browser_safe_mp4_replaces_overlay_with_converted_file(tmp_path: Path, monkeypatch) -> None:
    overlay = tmp_path / "pose_overlay.mp4"
    overlay.write_bytes(b"mp4v")

    monkeypatch.setattr(pose.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert "-c:v" in command
        assert "libx264" in command
        assert "-pix_fmt" in command
        assert "yuv420p" in command
        Path(command[-1]).write_bytes(b"h264")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pose.subprocess, "run", fake_run)

    assert pose._make_browser_safe_mp4(overlay)
    assert overlay.read_bytes() == b"h264"


def test_make_browser_safe_mp4_skips_when_ffmpeg_missing(tmp_path: Path, monkeypatch) -> None:
    overlay = tmp_path / "pose_overlay.mp4"
    overlay.write_bytes(b"mp4v")
    monkeypatch.setattr(pose.shutil, "which", lambda name: None)

    assert not pose._make_browser_safe_mp4(overlay)
    assert overlay.read_bytes() == b"mp4v"
