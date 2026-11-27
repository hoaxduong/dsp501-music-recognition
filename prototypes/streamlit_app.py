"""Streamlit interface for the DSP501 fingerprint experiment."""

import io
import os

import numpy as np
import soundfile as sf
import streamlit as st
from audio_input import MAX_BYTES, MAX_SECONDS, decode_audio
from dsp_steps import show_dsp_steps, show_offsets
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
from loudio_catalog import LOUDIO_DATA, load_loudio_references
from loudio_fingerprints import EXAMPLE_FILE, INDEX_FILE, CompactIndex, export_query

APP_TITLE = "Music Recording Identification"


@st.cache_resource(show_spinner="Building the reference index…")
def catalog(use_loudio):
    if use_loudio:
        references, titles = load_loudio_references()
    else:
        references = load_references()
        titles = {key: RECORDINGS[key][0] for key in references}
    return references, build_index(references), titles


def wav_bytes(wave):
    output = io.BytesIO()
    gain = min(1, 0.95 / max(np.max(np.abs(wave)), 1e-8))
    sf.write(output, wave * gain, SR, format="WAV")
    return output.getvalue()


def show_match(result, duration):
    match = result["match"]
    if match:
        st.success(f"{match['title']} · estimated start {max(0, match['seconds']):g} s")
    else:
        st.info("No reliable match.")
    st.caption(f"{duration:.2f} s · {result['tokens']:,} peak pairs in the query")
    candidates = result["candidates"]
    if len(candidates) > 10:
        st.caption(f"Top 10 candidates out of {len(candidates)} recordings.")
    st.table([{"Recording": item["title"], "Votes": item["votes"]} for item in candidates[:10]])
    st.caption("A match needs at least 8 votes and at least twice the runner-up score.")


def show_result(wave, index, keys, titles):
    result = recognize(wave, index, keys, explain=True, titles=titles)
    show_match(result, len(wave) / SR)
    show_dsp_steps(wave, result)


def download_fingerprint(wave):
    st.download_button(
        "Export query fingerprint",
        export_query(wave),
        file_name="query-fingerprint.json",
        mime="application/json",
        on_click="ignore",
        help="Send this fingerprint file to the cloud demo. It contains no audio samples.",
    )


@st.cache_resource(show_spinner="Loading the Loudio fingerprint index…")
def cloud_index():
    return CompactIndex(INDEX_FILE)


def show_loudio_cloud():
    index = cloud_index()
    st.caption(f"{len(index.titles)} Loudio recordings · first {REFERENCE_SECONDS} seconds each")
    st.write(
        "Use a fingerprint exported from the local app. Only peak-pair keys and anchor positions "
        "are sent to this service. For a quick test, use the example fingerprint."
    )
    query_file = st.file_uploader("Upload query fingerprint", type=["json"], max_upload_size=1)
    example = st.button("Try an example fingerprint")
    if not example and (query_file is None or not st.button("Identify recording", type="primary")):
        return
    payload = EXAMPLE_FILE.read_bytes() if example else query_file.getvalue()
    try:
        seconds, result = index.recognize(payload)
    except ValueError as error:
        st.error(str(error))
        return
    show_match(result, seconds)
    show_offsets(result["trace"], result["candidates"][0])


def main():
    st.set_page_config(page_title=APP_TITLE)
    st.title(APP_TITLE)
    st.write("Identify short audio clips using spectral peak pairs and time-offset voting.")
    use_loudio = os.getenv("DSP501_LOCAL_LOUDIO") == "1"
    if not use_loudio and INDEX_FILE.is_file():
        reference_catalog = st.radio(
            "Reference catalog", ("Loudio fingerprints", "Wikimedia audio demo"), horizontal=True
        )
        if reference_catalog == "Loudio fingerprints":
            show_loudio_cloud()
            return
    try:
        references, index, titles = catalog(use_loudio)
    except (OSError, ValueError, RuntimeError) as error:
        # Wikimedia or its network route may be unavailable on a cold start.
        if use_loudio:
            st.error(f"Could not load the local Loudio dataset at {LOUDIO_DATA}.")
        else:
            st.error("Could not load the reference recordings. Try refreshing the page.")
        print(f"Error loading reference recordings: {error}", flush=True)
        return

    mode = st.radio("Audio source", ("Sample clip", "Upload", "Microphone"), horizontal=True)
    keys = tuple(references)
    source = "Local Loudio dataset" if use_loudio else "Wikimedia Commons catalog"
    st.caption(f"{source}: {len(keys)} recordings, first {REFERENCE_SECONDS} seconds of each.")

    if mode == "Sample clip":
        song = st.selectbox("Recording", keys, index=0 if use_loudio else 1, format_func=titles.get)
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
        if use_loudio:
            download_fingerprint(wave)
        if st.button("Identify sample", type="primary"):
            show_result(
                wave, index, tuple(key for key in keys if not exclude or key != song), titles
            )
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
        location = "this computer" if use_loudio else "the server"
        st.caption(f"The app analyzes the first {MAX_SECONDS} seconds on {location}.")
        if audio is not None and st.button("Identify recording", type="primary"):
            try:
                if audio.size > MAX_BYTES:
                    raise ValueError("The audio exceeds the 20 MB limit.")
                wave = decode_audio(audio.getvalue())
                show_result(wave, index, keys, titles)
                if use_loudio:
                    download_fingerprint(wave)
            except ValueError as error:
                st.error(str(error))

    if use_loudio:
        return
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
