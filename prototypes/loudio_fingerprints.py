"""Share peak-pair digests and frame offsets without sending audio."""

import json
import re
from collections import Counter
from functools import lru_cache
from hashlib import blake2b
from pathlib import Path
from struct import pack

import numpy as np
from audio_input import MAX_SECONDS
from fingerprint import FRAME, HOP, SR, fingerprints, match_tokens, spectrum

FORMAT = "dsp501-peak-pairs-v1"
ASSETS = Path(__file__).with_name("fingerprint_assets")
INDEX_FILE = ASSETS / "loudio-index.npz"
EXAMPLE_FILE = ASSETS / "example-query.json"
MAX_QUERY_BYTES = 500_000
MAX_VOTES_TO_CHECK = 1_000_000


@lru_cache(maxsize=131_072)
def token_digest(token):
    return int.from_bytes(
        blake2b(pack("<HHH", *token), digest_size=8, person=b"dsp501-v1").digest()
    )


def export_query(wave):
    if not np.isfinite(wave).all() or not SR <= len(wave) <= MAX_SECONDS * SR:
        raise ValueError(f"Provide between 1 and {MAX_SECONDS} seconds of valid audio.")
    pairs = fingerprints(np.abs(spectrum(wave)[2]))
    return json.dumps(
        {
            "format": FORMAT,
            "samples": len(wave),
            "pairs": [[f"{token_digest(token):016x}", frame] for token, frame in pairs],
        },
        separators=(",", ":"),
    ).encode()


def load_query(payload):
    if len(payload) > MAX_QUERY_BYTES:
        raise ValueError("The fingerprint file is too large.")
    try:
        data = json.loads(payload)
        samples = data["samples"]
        if data["format"] != FORMAT or type(samples) is not int:
            raise ValueError
        if not SR <= samples <= MAX_SECONDS * SR:
            raise ValueError
        pairs = data["pairs"]
        last_frame = (samples - FRAME) // HOP
        if not isinstance(pairs, list) or len(pairs) > 16_000:
            raise ValueError
        query = []
        for digest, frame in pairs:
            if not re.fullmatch(r"[0-9a-f]{16}", digest):
                raise ValueError
            if type(frame) is not int or not 0 <= frame <= last_frame:
                raise ValueError
            query.append((int(digest, 16), frame))
        if any(count > 8 for count in Counter(query).values()):
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError) as error:
        raise ValueError(
            "Invalid fingerprint file. Export a new JSON query from the local app."
        ) from error
    return samples / SR, query


class CompactIndex:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            if data["format"].item() != FORMAT:
                raise ValueError("The fingerprint index version does not match the query format.")
            self.digests = data["digests"]
            self.starts = data["starts"]
            self.postings = data["postings"]
            self.titles = dict(enumerate(data["titles"].tolist()))

    def get(self, digest, default):
        idx = self.digests.searchsorted(np.uint64(digest))
        if idx == len(self.digests) or self.digests[idx] != digest:
            return default
        return self.postings[self.starts[idx] : self.starts[idx + 1]].tolist()

    def recognize(self, payload):
        seconds, query = load_query(payload)
        digests = np.asarray([digest for digest, _ in query], dtype=np.uint64)
        positions = self.digests.searchsorted(digests)
        in_range = positions < len(self.digests)
        positions = positions[in_range]
        positions = positions[self.digests[positions] == digests[in_range]]
        if (self.starts[positions + 1] - self.starts[positions]).sum() > MAX_VOTES_TO_CHECK:
            raise ValueError(
                "This fingerprint has too many repeated matches. Export a shorter clip."
            )
        return seconds, match_tokens(query, self, self.titles, titles=self.titles, explain=True)
