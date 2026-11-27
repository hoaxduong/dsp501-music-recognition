"""Dấu vân tay âm thanh từ notebooks/02-tham-tu-am-nhac.ipynb."""

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pooch
import soundfile as sf
from scipy.ndimage import maximum_filter
from scipy.signal import resample_poly, stft

SR = 12_000
FRAME, HOP = 1024, 240
WINDOW = "hann"
REFERENCE_SECONDS = 45
SEED = 501
CACHE = Path(__file__).resolve().parents[1] / "data/raw/audio-fingerprints"

# Giữ nguyên cửa sổ STFT, cách chọn đỉnh và tạo khóa của notebook.
fft_frequencies = np.fft.rfftfreq(FRAME, 1 / SR)

BASE_URL = "https://upload.wikimedia.org/wikipedia/commons/"
RECORDINGS = {
    "A": ("Für Elise", "fur-elise.ogg", "7/7b/FurElise.ogg"),
    "B": (
        "Jingle Bells",
        "jingle-bells.ogg",
        "f/f0/Jingle_Bells_%2890bpm%29_%28Kevin_MacLeod%29_%28ISRC_USUAN1100187%29.oga",
    ),
    "C": (
        "Maple Leaf Rag",
        "maple-leaf-rag.ogg",
        "3/3a/1906_-_Scott_Joplin%27s_Maple_Leaf_Rag_%281899%29_played_by_the_United_States_Marine_Band.ogg",
    ),
    "D": (
        "In the Hall of the Mountain King",
        "mountain-king.ogg",
        "b/bb/Musopen_-_In_the_Hall_Of_The_Mountain_King.ogg",
    ),
    "E": (
        "Symphony No. 5",
        "symphony-5.ogg",
        "e/e6/Ludwig_van_Beethoven_-_symphony_no._5_in_c_minor%2C_op._67_-_i._allegro_con_brio.ogg",
    ),
}
FILE_HASHES = {
    "fur-elise.ogg": "8deefb57df989a2b53a6bdd3e59813b6c34d61dca666caa39e53fa9597b378e3",
    "jingle-bells.ogg": "67216c89aa23862960d51d4593c6f3a7a00b34f6a0f525589299ebb3f2847080",
    "maple-leaf-rag.ogg": "90ed6bbb9554a5d6ffc63f599704dc185209e4155eaef1d501554a468ee32095",
    "mountain-king.ogg": "3bfa7424d3a148f1482a44688b77eb9138a7cea6cff7474799f1329181ab3342",
    "symphony-5.ogg": "663cfa62707be850a51ec8cd7b51b95595a9ee010f3eefea16e1dd46d54a9d95",
}


def spectrum(wave):
    return stft(
        wave,
        fs=SR,
        window=WINDOW,
        nperseg=FRAME,
        noverlap=FRAME - HOP,
        boundary=None,
        padded=False,
    )


def spectral_peaks(frequencies, magnitude):
    nearby = maximum_filter(magnitude, size=(11, 13))
    audible_band = (frequencies >= 120) & (frequencies <= 3500)
    strong = (magnitude == nearby) & (magnitude > 0.018 * magnitude.max())
    frequency_bins, frames = np.nonzero(strong & audible_band[:, None])
    return sorted(zip(frames.tolist(), frequency_bins.tolist()))


def fingerprints(magnitude):
    peaks = spectral_peaks(fft_frequencies, magnitude)
    for token, anchor, _ in fingerprint_pairs(peaks):
        yield token, anchor[0]


def fingerprint_pairs(peaks):
    for index, (anchor_time, anchor_freq) in enumerate(peaks):
        paired = 0
        for target in range(index + 1, len(peaks)):
            target_time, target_freq = peaks[target]
            delay = target_time - anchor_time
            if delay > 45:
                break
            if delay >= 5:
                token = (anchor_freq // 2, target_freq // 2, round(delay / 2))
                yield token, (anchor_time, anchor_freq), (target_time, target_freq)
                paired += 1
                if paired == 8:
                    break


def with_noise(wave, snr_db, rng):
    noise = rng.standard_normal(len(wave))
    target_rms = np.sqrt(np.mean(wave**2)) / 10 ** (snr_db / 20)
    return wave + noise * target_rms / np.sqrt(np.mean(noise**2))


def read_audio(source, *, seconds=15):
    with sf.SoundFile(source) as recording:
        rate = recording.samplerate
        samples = recording.read(int(seconds * rate), dtype="float32", always_2d=True)
    ratio = Fraction(SR, rate)
    return resample_poly(samples.mean(axis=1), ratio.numerator, ratio.denominator)


def load_references():
    downloader = pooch.HTTPDownloader(
        headers={"User-Agent": "DSP501 classroom experiment"}, timeout=60
    )
    references = {}
    for key, (_, filename, remote) in RECORDINGS.items():
        path = pooch.retrieve(
            BASE_URL + remote,
            known_hash=FILE_HASHES[filename],
            fname=filename,
            path=CACHE,
            downloader=downloader,
        )
        wave = read_audio(path, seconds=REFERENCE_SECONDS)
        references[key] = wave / np.max(np.abs(wave))
    return references


def build_index(references):
    index = defaultdict(list)
    for key, wave in references.items():
        for token, frame in fingerprints(np.abs(spectrum(wave)[2])):
            index[token].append((key, frame))
    return index


def recognize(wave, index, keys, *, explain=False, titles=None):
    if not np.isfinite(wave).all() or len(wave) < SR:
        raise ValueError("Provide at least one second of valid audio.")
    frequencies, times, coefficients = spectrum(wave)
    magnitude = np.abs(coefficients)
    peaks = spectral_peaks(frequencies, magnitude)
    pairs = list(fingerprint_pairs(peaks))
    query = [(token, anchor[0]) for token, anchor, _ in pairs]
    result = match_tokens(query, index, keys, explain=explain, titles=titles)
    if explain:
        result["trace"].update(
            frequencies=frequencies, times=times, magnitude=magnitude, peaks=peaks, pairs=pairs[:6]
        )
    return result


def match_tokens(query, index, keys, *, explain=False, titles=None):
    offsets = {key: Counter() for key in keys}
    for token, query_time in query:
        for key, ref_time in index.get(token, ()):
            if key in offsets:
                offsets[key][ref_time - query_time] += 1

    candidates = []
    for key, counts in offsets.items():
        votes, offset = max(
            (
                (hits + counts[shift - 1] + counts[shift + 1], shift)
                for shift, hits in counts.items()
            ),
            key=lambda pair: (pair[0], counts[pair[1]]),
            default=(0, None),
        )
        candidates.append(
            {
                "key": key,
                "title": RECORDINGS[key][0] if titles is None else titles[key],
                "votes": votes,
                "seconds": round(offset * HOP / SR, 2) if offset is not None else None,
            }
        )
    candidates.sort(key=lambda match: match["votes"], reverse=True)
    best, runner_up = candidates[:2]
    # Ngưỡng thử nghiệm cho kho nhỏ, dùng chung cho giao diện và phép đo.
    accepted = best["votes"] >= max(8, runner_up["votes"] * 2)
    result = {"match": best if accepted else None, "candidates": candidates, "tokens": len(query)}
    if explain:
        result["trace"] = {"offsets": offsets[best["key"]]}
    return result
