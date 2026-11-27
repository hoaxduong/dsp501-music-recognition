# Music Recording Identification

Identify snippets by matching spectral peak pairs and voting on time offsets.

[Open the Streamlit demo](https://music-recognition.streamlit.app/).

```bash
uv run --python 3.11 --with-requirements prototypes/requirements.txt streamlit run prototypes/streamlit_app.py
```

Streamlit Community Cloud uses `prototypes/streamlit_app.py` as the entry point.

## Identify a recording

Choose **Upload audio** or **Record audio**. Matching starts automatically after
the clip is ready. The browser analyzes up to 15 seconds of an uploaded clip
or records five seconds from the microphone. Audio stays in the browser. Only
the peak-pair fingerprints go to the Streamlit matcher.

The default Loudio index covers the first 45 seconds of 100 recordings. It
contains peak-pair digests, frame offsets, and display names. The Loudio
reference audio and dataset manifests stay on the local machine.

## Notebook thí nghiệm

[Nhận diện bản thu âm nhạc từ đoạn âm thanh ngắn](notebooks/02-tham-tu-am-nhac.ipynb)
so sánh cosine, 3-NN, DTW và dấu vân tay phổ ở ba mức nhiễu. Notebook có sẵn
bảng kết quả và biểu đồ. Chạy JupyterLab từ thư mục repo để thử lại với năm
bản thu đi kèm.

```bash
uv run --python 3.11 --with jupyterlab --with matplotlib --with pandas \
  --with-requirements prototypes/requirements.txt jupyter lab notebooks/02-tham-tu-am-nhac.ipynb
```

## DSP method and evaluation

The app identifies recordings using STFT, spectral peak pairs, and time-offset
voting. A match needs at least eight votes and at least twice the runner-up
score.

The Loudio dataset is kept local and was used for the reported experiments.
The [training and evaluation notebook](notebooks/04-loudio-match-training.ipynb)
records the separate decision-model experiment. Re-running it requires the
local Loudio data under `data/processed/loudio-hits/`.

## Notebook recording sources

The five Wikimedia Commons recordings below are used by the introductory notebook.

- Für Elise · V Gao (Gaodifan) · CC0. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:FurElise.ogg)
- Jingle Bells · Kevin MacLeod (incompetech.com) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jingle_Bells.ogg)
- Maple Leaf Rag · United States Marine Band. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:1906_-_Scott_Joplin%27s_Maple_Leaf_Rag_%281899%29_played_by_the_United_States_Marine_Band.ogg)
- In the Hall of the Mountain King · Czech National Symphony Orchestra · Musopen. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Musopen_-_In_the_Hall_Of_The_Mountain_King.ogg)
- Giao hưởng số 5 · Skidmore College Orchestra · Musopen. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ludwig_van_Beethoven_-_symphony_no._5_in_c_minor%2C_op._67_-_i._allegro_con_brio.ogg)
