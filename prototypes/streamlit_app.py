"""Identify audio against the Loudio index using fingerprints from the browser."""

import json
from pathlib import Path

import streamlit as st
from fingerprint import REFERENCE_SECONDS
from loudio_fingerprints import INDEX_FILE, CompactIndex

APP_TITLE = "Music Recording Identification"

browser_audio = st.components.v2.component(
    "dsp501_browser_audio",
    html="""
        <div class="choices">
            <label>
                <span>Upload audio</span>
                <input type="file" accept="audio/*,.m4a,.webm,.mp4">
            </label>
            <button type="button">Record audio</button>
        </div>
        <p role="status" aria-live="polite"></p>
    """,
    css="""
        .choices {
            display: flex;
            align-items: end;
            gap: 1rem;
            flex-wrap: wrap;
        }
        label { display: grid; gap: 0.35rem; }
        input, button { font: inherit; color: inherit; max-width: 100%; }
        button {
            cursor: pointer;
            border: 1px solid var(--st-border-color, #888);
            padding: 0.4rem 0.8rem;
            border-radius: 0.5rem;
            background: var(--st-secondary-background-color, transparent);
        }
        button:disabled { cursor: wait; }
        p { margin: 0.45rem 0 0; }
    """,
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
    st.set_page_config(page_title=APP_TITLE)
    st.title(APP_TITLE)
    try:
        index = loudio_index()
    except (OSError, ValueError):
        st.error("The recording index is unavailable.")
        return

    st.caption(f"{len(index.titles)} recordings indexed · first {REFERENCE_SECONDS} s each")
    st.write("Choose an audio clip or record five seconds to identify the recording.")
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

    match = result["match"]
    if match:
        start = max(0, match["seconds"])
        st.success(match["title"])
        st.caption(f"Estimated start in reference {start:g} s")
    else:
        st.info("No reliable match in the catalog.")
    st.caption(f"Analyzed {seconds:.2f} s · {result['tokens']:,} spectral peak pairs")


if __name__ == "__main__":
    main()
