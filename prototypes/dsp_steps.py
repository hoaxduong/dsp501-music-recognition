"""Display the intermediate DSP values actually used by the recognizer."""

from threading import Lock

import numpy as np
import streamlit as st
from fingerprint import FRAME, HOP, SR, WINDOW
from matplotlib.figure import Figure
from scipy.signal import get_window

PLOT_LOCK = Lock()
ACCENT = "#21726a"
HIGHLIGHT = "#a33d2c"


def show_signal(wave, frame):
    st.markdown("**DSP501 · Discrete-time signal and sampling**")
    st.write(
        f"The app converts audio to one channel at {SR:,} samples per second. "
        "The top plot shows the processed clip. The bottom plot zooms in on one frame "
        "and shows how the Hann window tapers both ends."
    )
    start = frame * HOP
    raw = wave[start : start + FRAME]
    tapered = raw * get_window(WINDOW, FRAME)
    with PLOT_LOCK:
        fig = Figure(figsize=(8.2, 4.5), layout="constrained")
        overview, zoom = fig.subplots(2)
        stride = max(1, len(wave) // 4000)
        overview.plot(np.arange(0, len(wave), stride) / SR, wave[::stride], lw=0.8, color=ACCENT)
        overview.axvspan(start / SR, (start + FRAME) / SR, color=HIGHLIGHT, alpha=0.22)
        overview.set(xlabel="Time within query (s)", ylabel="Amplitude", xlim=(0, len(wave) / SR))
        ms = np.arange(FRAME) * 1000 / SR
        zoom.plot(ms, raw, label="Before windowing", color="#64748b", lw=0.9)
        zoom.plot(ms, tapered, label="After Hann windowing", color=ACCENT, lw=1.2)
        zoom.set(xlabel="Time within frame (ms)", ylabel="Amplitude")
        zoom.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")


def show_fourier(trace, frame):
    st.markdown("**DSP501 · DFT, FFT, and STFT**")
    st.write(
        f"An FFT computes the DFT of each {FRAME}-sample frame "
        f"({1000 * FRAME / SR:.1f} ms). Frames advance by {HOP} samples "
        f"({1000 * HOP / SR:.0f} ms). Frequency bins are {SR / FRAME:.2f} Hz apart. "
        "The spectrogram shows how the spectrum changes over time."
    )
    st.latex(r"X_m[k] = \sum_{n=0}^{N-1} x[mH+n]\,w[n]\,e^{-j2\pi kn/N}")
    st.caption("N is the frame length in samples, H is the hop size, and w[n] is the Hann window.")
    times, hz, magnitude = (trace[name] for name in ("times", "frequencies", "magnitude"))
    visible = hz <= 3500
    maximum = float(magnitude.max()) or 1.0
    db = 20 * np.log10(np.maximum(magnitude[visible] / maximum, 1e-4))
    with PLOT_LOCK:
        fig = Figure(figsize=(8.2, 5.2), layout="constrained")
        single, spectrogram = fig.subplots(2)
        single.plot(hz[visible], db[:, frame], color=ACCENT, lw=1)
        single.set(
            title=f"Spectrum of the frame at {times[frame]:.3f} s",
            xlabel="Frequency (Hz)",
            ylabel="Level (dB)",
            ylim=(-80, 3),
        )
        mesh = spectrogram.pcolormesh(
            times, hz[visible], db, shading="auto", cmap="magma", vmin=-60, vmax=0
        )
        spectrogram.axvline(times[frame], color="#9eece0", lw=1, linestyle="--")
        if trace["peaks"]:
            frames, bins = np.asarray(trace["peaks"]).T
            spectrogram.scatter(
                times[frames],
                hz[bins],
                facecolors="none",
                edgecolors="#9eece0",
                s=9,
                linewidths=0.6,
            )
        if trace["pairs"]:
            _, (start, f1), (stop, f2) = trace["pairs"][0]
            spectrogram.plot(
                times[[start, stop]], hz[[f1, f2]], color="white", marker="o", ms=4, lw=1.2
            )
        spectrogram.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 3500))
        fig.colorbar(mesh, ax=spectrogram, label="Relative level (dB)")
        st.pyplot(fig, width="stretch")
    st.caption(
        "Circles mark selected peaks. The white line joins the first peak pair when one is available."
    )


def show_pairs(trace, tokens):
    st.markdown("**Fingerprinting by spectral peak pairing**")
    st.write(
        f"The app selected **{len(trace['peaks']):,} peaks** between 120 and 3,500 Hz and formed "
        f"**{tokens:,} pairs**. Each key encodes two frequency bins and the time between them. "
        "The table shows pairs from this query."
    )
    if not trace["pairs"]:
        st.info("This clip produced no usable peak pairs.")
        return
    hz = trace["frequencies"]
    st.table(
        [
            {
                "Peak 1 (Hz)": round(hz[f1], 1),
                "Peak 2 (Hz)": round(hz[f2], 1),
                "Separation (ms)": round((stop - start) * HOP * 1000 / SR),
                "Key": str(token),
            }
            for token, (start, f1), (stop, f2) in trace["pairs"]
        ]
    )


def show_offsets(trace, leading):
    st.markdown("**Matching by time-offset voting**")
    st.write(
        "Each matching key votes for its reference time minus its query time. "
        "Votes that cluster around one offset support the same "
        "recording and start time."
    )
    if not trace["offsets"]:
        st.info("No matching votes to plot.")
        return
    shifts = np.array(sorted(trace["offsets"]))
    votes = [trace["offsets"][shift] for shift in shifts]
    center = leading["seconds"]
    with PLOT_LOCK:
        fig = Figure(figsize=(8.2, 3), layout="constrained")
        ax = fig.subplots()
        ax.vlines(shifts * HOP / SR, 0, votes, color=ACCENT, lw=1)
        ax.axvline(center, color=HIGHLIGHT, linestyle="--", label=f"Winning offset ({center:g} s)")
        ax.set(
            title=leading["title"],
            xlabel="Time offset (s)",
            ylabel="Votes per offset bin",
            ylim=(0, None),
        )
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")
    st.caption(
        "The table score sums the votes at the red line and its two neighboring bins (±20 ms)."
    )


def show_dsp_steps(wave, result):
    trace = result["trace"]
    frame = int(np.unravel_index(trace["magnitude"].argmax(), trace["magnitude"].shape)[1])
    st.subheader("Signal processing steps")
    signal, fourier, pairs, offsets = st.tabs(["Signal", "DFT / STFT", "Fingerprint", "Matching"])
    with signal:
        show_signal(wave, frame)
    with fourier:
        show_fourier(trace, frame)
    with pairs:
        show_pairs(trace, result["tokens"])
    with offsets:
        show_offsets(trace, result["candidates"][0])
