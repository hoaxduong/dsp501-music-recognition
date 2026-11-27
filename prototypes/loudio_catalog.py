"""Load the processed Loudio references from a local repository."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from fingerprint import REFERENCE_SECONDS, SR

LOUDIO_DATA = Path(__file__).resolve().parents[1] / "data/processed/loudio-hits"


def load_loudio_references():
    references, titles = {}, {}
    with (LOUDIO_DATA / "tracks.csv").open(encoding="utf-8", newline="") as manifest:
        for row in csv.DictReader(manifest):
            key = row["song_id"]
            path = (LOUDIO_DATA / row["reference_file"]).resolve()
            if key in references or not path.is_relative_to(LOUDIO_DATA / "references"):
                raise ValueError("Invalid or duplicate reference in the Loudio manifest.")
            wave, rate = sf.read(path, dtype="float32")
            if rate != SR or wave.shape != (REFERENCE_SECONDS * SR,) or not np.isfinite(wave).all():
                raise ValueError(f"Invalid reference audio for {row['title']}.")
            references[key] = wave
            titles[key] = f"{row['title']} — {row['artist']}" if row["artist"] else row["title"]
    if len(references) < 3:
        raise ValueError("The Loudio catalog needs at least three reference recordings.")
    return references, titles
