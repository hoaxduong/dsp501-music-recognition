# Music Recording Identification

A DSP501 project that matches short excerpts to indexed recordings using
spectral peak pairs and time-offset voting.

- [Open the Streamlit app](https://music-recognition.streamlit.app/)
- [Read or run the experiment notebook](notebooks/music-recording-identification.ipynb)
- [Inspect the aggregate holdout counts](results/holdout_metrics.csv)

## Run locally

Use Python 3.11 for the app and notebook environment.

```bash
uv run --python 3.11 --with-requirements prototypes/requirements.txt \
  streamlit run prototypes/streamlit_app.py
```

Choose **Upload audio** or **Record audio** under **Audio input**. Uploads use
up to 15 seconds. Microphone recordings can run for 5, 10, or 15 seconds.
The browser processes the audio and displays the DSP steps. The server
receives spectral peak-pair fingerprints for matching. An accepted result
shows the recording and its estimated start time.

## Reproduce the study

```bash
uv run --python 3.11 --with-requirements prototypes/requirements.txt \
  --with jupyterlab --with pandas jupyter lab notebooks/music-recording-identification.ipynb
```

The notebook runs its signal-processing example and results from the included
aggregate counts. To repeat the full 180-query evaluation, set
`DSP501_DATASET` to the original processed corpus directory containing
`tracks.csv`, `queries.csv`, `references/`, and `queries/`. The default is
`data/processed/loudio-hits/` inside the checkout.

The demo index contains 100 recordings and covers their first 45 seconds.
The open-set experiment indexes 90 recordings and uses five-second test
excerpts. It identified 120 of 135 indexed queries and rejected all 45
withheld queries. The five withheld recordings and simulated noise define
the scope of those results.

## Repository layout

| Path | Purpose |
| --- | --- |
| [Notebook](notebooks/music-recording-identification.ipynb) | DSP method, experiment, results, and local replay |
| [Results](results/holdout_metrics.csv) | Fixed-rule counts by noise condition |
| [App entry point](prototypes/streamlit_app.py) | Streamlit UI and server-side matching |
| [Browser DSP](prototypes/browser_fingerprint.mjs) | Local audio decoding, fingerprints, and signal plots |
| [Reference index](prototypes/fingerprint_assets/loudio-index.npz) | Peak-pair digests, frame offsets, and display titles |

Streamlit Community Cloud uses `prototypes/streamlit_app.py` as the entry
point. The source audio, manifests, and generated query clips stay in the
local dataset directory.
