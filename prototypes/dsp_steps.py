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
    st.markdown("**DSP501 · Tín hiệu rời rạc và lấy mẫu**")
    st.write(
        f"Âm thanh được đưa về một kênh ở {SR:,} mẫu/giây. "
        "Đồ thị trên cho thấy đoạn đã xử lý. Đồ thị dưới phóng to một khung và "
        "cho thấy cửa sổ Hann làm nhẹ biên độ ở hai đầu khung."
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
        overview.set(xlabel="Thời gian trong đoạn (s)", ylabel="Biên độ", xlim=(0, len(wave) / SR))
        ms = np.arange(FRAME) * 1000 / SR
        zoom.plot(ms, raw, label="Mẫu trong khung", color="#64748b", lw=0.9)
        zoom.plot(ms, tapered, label="Sau cửa sổ Hann", color=ACCENT, lw=1.2)
        zoom.set(xlabel="Thời gian trong khung (ms)", ylabel="Biên độ")
        zoom.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")


def show_fourier(trace, frame):
    st.markdown("**DSP501 · DFT, FFT và STFT**")
    st.write(
        f"App tính DFT của từng khung bằng FFT. Mỗi khung có {FRAME} mẫu "
        f"({1000 * FRAME / SR:.1f} ms), dịch {HOP} mẫu ({1000 * HOP / SR:.0f} ms) "
        f"mỗi lần. Khoảng cách giữa hai ô tần số là {SR / FRAME:.2f} Hz. "
        "Xếp các phổ theo thứ tự thời gian sẽ thu được phổ thời gian–tần số."
    )
    st.latex(r"X_m[k] = \sum_{n=0}^{N-1} x[mH+n]\,w[n]\,e^{-j2\pi kn/N}")
    st.caption("N là số mẫu trong một khung, H là bước dịch và w[n] là cửa sổ Hann.")
    times, hz, magnitude = (trace[name] for name in ("times", "frequencies", "magnitude"))
    visible = hz <= 3500
    maximum = float(magnitude.max()) or 1.0
    db = 20 * np.log10(np.maximum(magnitude[visible] / maximum, 1e-4))
    with PLOT_LOCK:
        fig = Figure(figsize=(8.2, 5.2), layout="constrained")
        single, spectrogram = fig.subplots(2)
        single.plot(hz[visible], db[:, frame], color=ACCENT, lw=1)
        single.set(
            title=f"Phổ của một khung tại {times[frame]:.3f} s",
            xlabel="Tần số (Hz)",
            ylabel="Mức (dB)",
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
        spectrogram.set(xlabel="Thời gian (s)", ylabel="Tần số (Hz)", ylim=(0, 3500))
        fig.colorbar(mesh, ax=spectrogram, label="dB tương đối")
        st.pyplot(fig, width="stretch")
    st.caption("Vòng tròn đánh dấu đỉnh phổ được giữ. Đường trắng nối cặp đỉnh đầu tiên nếu có.")


def show_pairs(trace, tokens):
    st.markdown("**Dấu vân tay · Ghép đỉnh phổ**")
    st.write(
        f"App giữ **{len(trace['peaks']):,} đỉnh** trong dải 120–3500 Hz và tạo "
        f"**{tokens:,} cặp**. Mỗi khóa mã hóa ô tần số của hai đỉnh và khoảng thời gian "
        "giữa chúng. Các cặp dưới đây lấy trực tiếp từ đoạn vừa phân tích."
    )
    if not trace["pairs"]:
        st.info("Đoạn này không tạo được cặp đỉnh phù hợp.")
        return
    hz = trace["frequencies"]
    st.table(
        [
            {
                "Đỉnh 1 (Hz)": round(hz[f1], 1),
                "Đỉnh 2 (Hz)": round(hz[f2], 1),
                "Cách nhau (ms)": round((stop - start) * HOP * 1000 / SR),
                "Khóa": str(token),
            }
            for token, (start, f1), (stop, f2) in trace["pairs"]
        ]
    )


def show_offsets(trace, leading):
    st.markdown("**Đối sánh theo độ lệch thời gian**")
    st.write(
        "Mỗi cặp trùng khóa bỏ một phiếu cho độ lệch giữa vị trí trong bản thu và "
        "vị trí trong đoạn nghe. Cụm phiếu dày quanh cùng một độ lệch là dấu hiệu "
        "đoạn nghe đến từ bản thu đó."
    )
    if not trace["offsets"]:
        st.info("Không có phiếu khớp để vẽ.")
        return
    shifts = np.array(sorted(trace["offsets"]))
    votes = [trace["offsets"][shift] for shift in shifts]
    center = leading["seconds"]
    with PLOT_LOCK:
        fig = Figure(figsize=(8.2, 3), layout="constrained")
        ax = fig.subplots()
        ax.vlines(shifts * HOP / SR, 0, votes, color=ACCENT, lw=1)
        ax.axvline(center, color=HIGHLIGHT, linestyle="--", label=f"Độ lệch nổi bật ({center:g} s)")
        ax.set(
            title=leading["title"],
            xlabel="Độ lệch (s)",
            ylabel="Phiếu ở mỗi độ lệch",
            ylim=(0, None),
        )
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")
    st.caption("Điểm trong bảng cộng phiếu tại vạch đỏ và ở hai độ lệch lân cận (±20 ms).")


def show_dsp_steps(wave, result):
    trace = result["trace"]
    frame = int(np.unravel_index(trace["magnitude"].argmax(), trace["magnitude"].shape)[1])
    st.subheader("Các bước xử lý tín hiệu")
    signal, fourier, pairs, offsets = st.tabs(["Tín hiệu", "DFT / STFT", "Dấu vân tay", "Đối sánh"])
    with signal:
        show_signal(wave, frame)
    with fourier:
        show_fourier(trace, frame)
    with pairs:
        show_pairs(trace, result["tokens"])
    with offsets:
        show_offsets(trace, result["candidates"][0])
