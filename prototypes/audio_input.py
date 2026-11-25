"""Decode uploaded audio and browser recordings."""

import io
import subprocess

import numpy as np
import soundfile as sf
from fingerprint import SR, read_audio

MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 15


def decode_audio(body):
    try:
        return read_audio(io.BytesIO(body), seconds=MAX_SECONDS)
    except sf.LibsndfileError:
        # MediaRecorder có thể trả WebM hoặc MP4, tùy trình duyệt.
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "pipe",
                    "-i",
                    "pipe:0",
                    "-t",
                    str(MAX_SECONDS),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(SR),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                input=body,
                capture_output=True,
                check=True,
                timeout=8,
            )
        except FileNotFoundError as error:
            raise ValueError(
                "Định dạng này cần FFmpeg. Bạn có thể thử tệp WAV hoặc MP3."
            ) from error
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ValueError("Không đọc được âm thanh từ tệp này.") from error
        return np.frombuffer(result.stdout, dtype="<f4")
