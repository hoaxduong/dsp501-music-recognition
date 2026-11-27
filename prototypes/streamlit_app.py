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

APP_TITLE = "Music Recording Identification"


@st.cache_resource(show_spinner="Building the reference index…")
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
        st.success(f"{match['title']} · estimated start {max(0, match['seconds']):g} s")
    else:
        st.info("No reliable match.")
    st.caption(f"{len(wave) / SR:.2f} s · {result['tokens']:,} peak pairs in the query")
    st.table(
        [{"Recording": item["title"], "Votes": item["votes"]} for item in result["candidates"]]
    )
    st.caption("A match needs at least 8 votes and at least twice the runner-up score.")
    show_dsp_steps(wave, result)


def main():
    st.set_page_config(page_title=APP_TITLE)
    st.title(APP_TITLE)
    st.write(
        "Identify short audio clips using spectral peak pairs and time-offset voting. The index covers the first 45 seconds of five recordings."
    )
    try:
        references, index = catalog()
    except (OSError, ValueError, RuntimeError) as error:
        # Wikimedia or its network route may be unavailable on a cold start.
        st.error("Could not load the reference recordings. Try refreshing the page.")
        print(f"Error loading reference recordings: {error}", flush=True)
        return

    mode = st.radio("Audio source", ("Sample clip", "Upload", "Microphone"), horizontal=True)
    keys = tuple(references)

    if mode == "Sample clip":
        song = st.selectbox("Recording", keys, index=1, format_func=lambda key: RECORDINGS[key][0])
        left, right = st.columns(2)
        duration = left.slider("Duration (seconds)", 1, 10, 5)
        start = right.slider(
            "Start time (seconds)",
            0.0,
            float(REFERENCE_SECONDS - duration),
            12.137,
            0.001,
            format="%.3f",
        )
        levels = {"Clean": None, "+10 dB": 10, "−3 dB": -3}
        noise = st.selectbox("White noise (SNR)", tuple(levels))
        exclude = st.checkbox("Exclude this recording from the search")
        first = round(start * SR)
        wave = references[song][first : first + duration * SR]
        if levels[noise] is not None:
            wave = with_noise(wave, levels[noise], np.random.default_rng(SEED))
        st.audio(wav_bytes(wave))
        if st.button("Identify sample", type="primary"):
            show_result(wave, index, tuple(key for key in keys if not exclude or key != song))
    else:
        if mode == "Upload":
            audio = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "ogg", "flac", "m4a", "mp4", "webm"],
                max_upload_size=MAX_BYTES // (1024 * 1024),
            )
        else:
            st.caption("Play music near the microphone and record about five seconds.")
            audio = st.audio_input("Record audio")
        st.caption(
            f"The app analyzes the first {MAX_SECONDS} seconds. "
            "Audio is sent to the server for analysis."
        )
        if audio is not None and st.button("Identify recording", type="primary"):
            try:
                if audio.size > MAX_BYTES:
                    raise ValueError("The audio exceeds the 20 MB limit.")
                wave = decode_audio(audio.getvalue())
                show_result(wave, index, keys)
            except ValueError as error:
                st.error(str(error))

    with st.expander("Recording credits"):
        st.caption(
            "The sample clips are trimmed, converted to 12 kHz mono, and can be mixed with noise."
        )
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
