# Music Recording Identification

Identify snippets by matching spectral peak pairs and voting on time offsets.

[Open the Streamlit demo](https://music-recognition.streamlit.app/).

```bash
uv run --python 3.11 --with-requirements prototypes/requirements.txt streamlit run prototypes/streamlit_app.py
```

Streamlit Community Cloud dùng `prototypes/streamlit_app.py` làm entrypoint và Python 3.11.

## Xem các bước xử lý

Sau khi bấm **Identify sample** hoặc **Identify recording**, mục **Signal processing steps** cho thấy dạng sóng
rời rạc, tác dụng của cửa sổ Hann, phổ DFT của một khung và ảnh phổ STFT.
Hai tab cuối trình bày cặp đỉnh dùng làm dấu vân tay và số phiếu theo độ lệch
thời gian. Thử đổi mức nhiễu của cùng một đoạn mẫu để quan sát các bước này.

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
score. The app offers a Loudio fingerprint catalog and a five-recording Wikimedia audio demo.

The Loudio dataset is kept local and was used for the reported experiments.
The [training and evaluation notebook](notebooks/04-loudio-match-training.ipynb)
records the separate decision-model experiment. Re-running it requires the
local Loudio data under `data/processed/loudio-hits/`.

## Loudio lookup without uploading audio

1. In the public demo, select **Loudio fingerprints** and click **Try an example fingerprint**.
2. For your own query, run the app locally with the processed Loudio dataset and export a fingerprint:

   ```bash
   DSP501_LOCAL_LOUDIO=1 uv run --python 3.11 --with-requirements prototypes/requirements.txt \
     streamlit run prototypes/streamlit_app.py --server.address 127.0.0.1 --server.port 8502
   ```

3. Use **Sample clip**, **Upload**, or **Microphone** on the local app. For an uploaded or
   recorded clip, click **Identify recording** to reveal **Export query fingerprint**.
   Upload the exported JSON in **Loudio fingerprints** on the public demo.

The Loudio catalog indexes the first 45 seconds of each of 100 recordings.
The repository stores peak-pair digests, frame positions and display names.
The Loudio audio files and dataset manifests stay on the local machine.
The JSON query is also derived data. The cloud workflow can display votes
by time offset because it compares those digests and frame positions.

## Nguồn âm thanh

Các tệp Wikimedia Commons dưới đây được giữ nguyên để ứng dụng khởi động
không cần tải lại. Khi chạy, ứng dụng kiểm tra SHA-256 và dùng 45 giây đầu.

- Für Elise · V Gao (Gaodifan) · CC0. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:FurElise.ogg)
- Jingle Bells · Kevin MacLeod (incompetech.com) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jingle_Bells.ogg)
- Maple Leaf Rag · United States Marine Band. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:1906_-_Scott_Joplin%27s_Maple_Leaf_Rag_%281899%29_played_by_the_United_States_Marine_Band.ogg)
- In the Hall of the Mountain King · Czech National Symphony Orchestra · Musopen. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Musopen_-_In_the_Hall_Of_The_Mountain_King.ogg)
- Giao hưởng số 5 · Skidmore College Orchestra · Musopen. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ludwig_van_Beethoven_-_symphony_no._5_in_c_minor%2C_op._67_-_i._allegro_con_brio.ogg)
