"""Streamlit interface for the DSP501 fingerprint experiment."""

import io

import numpy as np
import soundfile as sf
import streamlit as st
from audio_input import MAX_BYTES, MAX_SECONDS, decode_audio
from dsp_steps import show_dsp_steps
from fingerprint import (
    RECORDINGS,
    REFERENCE_SECONDS,
    SEED,
    SR,
    build_index,
    load_references,
    recognize,
    with_noise,
)

APP_TITLE = "Nhận diện bản thu âm nhạc"


@st.cache_resource(show_spinner="Đang chuẩn bị kho 5 bản thu…")
def catalog():
    references = load_references()
    return references, build_index(references)


def wav_bytes(wave):
    output = io.BytesIO()
    gain = min(1, 0.95 / max(np.max(np.abs(wave)), 1e-8))
    sf.write(output, wave * gain, SR, format="WAV")
    return output.getvalue()


def show_result(wave, index, keys):
    result = recognize(wave, index, keys, explain=True)
    match = result["match"]
    if match:
        st.success(f"{match['title']} · khớp khoảng giây {max(0, match['seconds']):g}")
    else:
        st.info("Không đủ dấu hiệu khớp.")
    st.caption(f"{len(wave) / SR:.2f} giây · {result['tokens']} cặp đỉnh trong đoạn nghe")
    st.table([{"Bản thu": item["title"], "Điểm": item["votes"]} for item in result["candidates"]])
    st.caption("Báo khớp từ 8 điểm và khi bài dẫn đầu có ít nhất gấp đôi điểm bài kế tiếp.")
    show_dsp_steps(wave, result)


def main():
    st.set_page_config(page_title=APP_TITLE)
    st.title(APP_TITLE)
    st.write("Thử nhận diện bằng dấu vân tay từ 45 giây đầu của 5 bản thu trong notebook.")
    try:
        references, index = catalog()
    except (OSError, ValueError, RuntimeError) as error:
        # Wikimedia or its network route may be unavailable on a cold start.
        st.error("Chưa tải được kho bản thu. Hãy thử tải lại trang.")
        print(f"Lỗi tải kho bản thu: {error}", flush=True)
        return

    mode = st.radio("Nguồn âm thanh", ("Đoạn mẫu", "Tải tệp", "Micro"), horizontal=True)
    keys = tuple(references)

    if mode == "Đoạn mẫu":
        song = st.selectbox("Bản thu", keys, index=1, format_func=lambda key: RECORDINGS[key][0])
        left, right = st.columns(2)
        duration = left.slider("Độ dài (giây)", 1, 10, 5)
        start = right.slider(
            "Vị trí bắt đầu (giây)",
            0.0,
            float(REFERENCE_SECONDS - duration),
            12.137,
            0.001,
            format="%.3f",
        )
        levels = {"Đoạn sạch": None, "+10 dB": 10, "−3 dB": -3}
        noise = st.selectbox("Nhiễu trắng", tuple(levels))
        exclude = st.checkbox("Bỏ bản thu này khỏi kho khi dò")
        first = round(start * SR)
        wave = references[song][first : first + duration * SR]
        if levels[noise] is not None:
            wave = with_noise(wave, levels[noise], np.random.default_rng(SEED))
        st.audio(wav_bytes(wave))
        if st.button("Nhận diện đoạn mẫu", type="primary"):
            show_result(wave, index, tuple(key for key in keys if not exclude or key != song))
    else:
        if mode == "Tải tệp":
            audio = st.file_uploader(
                "Chọn tệp âm thanh",
                type=["wav", "mp3", "ogg", "flac", "m4a", "mp4", "webm"],
                max_upload_size=MAX_BYTES // (1024 * 1024),
            )
        else:
            st.caption("Bật nhạc gần micro, thu khoảng 5 giây rồi bấm dừng.")
            audio = st.audio_input("Thu âm")
        st.caption(
            f"Phân tích tối đa {MAX_SECONDS} giây đầu. "
            "Âm thanh được gửi tới máy chủ nhận diện trong phiên hiện tại."
        )
        if audio is not None and st.button("Nhận diện", type="primary"):
            try:
                if audio.size > MAX_BYTES:
                    raise ValueError("Đoạn âm thanh vượt giới hạn 20 MB.")
                wave = decode_audio(audio.getvalue())
                show_result(wave, index, keys)
            except ValueError as error:
                st.error(str(error))

    with st.expander("Nguồn bản thu"):
        st.caption("Các đoạn mẫu được cắt ngắn, chuyển sang mono 12 kHz và có thể được thêm nhiễu.")
        performers = {
            "A": "V Gao (Gaodifan)",
            "B": "Kevin MacLeod (incompetech.com) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)",
            "C": "United States Marine Band",
            "D": "Czech National Symphony Orchestra · Musopen",
            "E": "Skidmore College Orchestra · Musopen",
        }
        for key, (title, _, remote) in RECORDINGS.items():
            source = f"https://commons.wikimedia.org/wiki/File:{remote.split('/')[-1]}"
            st.markdown(f"- [{title}]({source}) · {performers[key]}")


if __name__ == "__main__":
    main()
