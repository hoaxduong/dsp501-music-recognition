"""Identify audio against the Loudio index using fingerprints from the browser."""

import json
from collections import Counter
from pathlib import Path

import streamlit as st
from fingerprint import REFERENCE_SECONDS
from loudio_fingerprints import INDEX_FILE, CompactIndex, load_query

APP_TITLE = "Music Recording Identification"

browser_audio = st.components.v2.component(
    "dsp501_browser_audio",
    html=Path(__file__).with_name("browser_audio.html").read_text(encoding="utf-8"),
    css=Path(__file__).with_name("browser_audio.css").read_text(encoding="utf-8"),
    js=Path(__file__).with_name("browser_fingerprint.mjs").read_text(encoding="utf-8"),
)


@st.cache_resource(show_spinner="Loading the recording index…")
def loudio_index():
    return CompactIndex(INDEX_FILE)


@st.fragment
def show_catalog(titles):
    st.subheader("Indexed recordings")
    st.caption(f"To test a recording, choose a clip from its first {REFERENCE_SECONDS} seconds.")
    search = st.text_input("Search by song or artist", placeholder="Enter a song or artist")
    terms = search.casefold().split()
    recordings = [
        {"Song and artist": title}
        for title in titles
        if all(term in title.casefold() for term in terms)
    ]

    st.caption(f"{len(recordings)} of {len(titles)} recordings")
    if recordings:
        st.dataframe(recordings, hide_index=True, width="stretch", height=320)
    else:
        st.info("No recordings match your search.")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    with st.container(horizontal_alignment="center"), st.container(width=1080):
        show_app()


def show_app():
    st.title(APP_TITLE)
    try:
        index = loudio_index()
    except (OSError, ValueError):
        st.error("The recording index is unavailable.")
        return

    st.caption(f"{len(index.titles)} recordings indexed · first {REFERENCE_SECONDS} s each")
    st.caption("Audio stays in your browser. Only fingerprints are sent for matching.")

    response = browser_audio(key="loudio_audio", on_query_change=lambda: None)
    if response.query:
        show_match(index, response.query)
    show_catalog(tuple(index.titles.values()))


def show_match(index, query):
    try:
        payload = json.dumps(query, separators=(",", ":")).encode()
        seconds, result = index.recognize(payload)
    except ValueError as error:
        st.error(str(error))
        return

    match_column, votes_column = st.columns([5, 7])
    with match_column:
        match = result["match"]
        if match:
            start = max(0, match["seconds"])
            st.success(match["title"])
            st.caption(f"Estimated start in reference {start:g} s")
        else:
            st.info("No reliable match in the catalog.")
        st.caption(f"Analyzed {seconds:.2f} s · {result['tokens']:,} spectral peak pairs")
    with votes_column, st.expander("See time-offset votes"):
        from dsp_steps import show_offsets

        leading = result["candidates"][0]
        _, pairs = load_query(payload)
        offsets = Counter(
            reference_frame - query_frame
            for digest, query_frame in pairs
            for recording, reference_frame in index.get(digest, ())
            if recording == leading["key"]
        )
        show_offsets({"offsets": offsets}, leading)


if __name__ == "__main__":
    main()
