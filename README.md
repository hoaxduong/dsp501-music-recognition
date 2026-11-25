# Thám tử âm nhạc

Web app thử nhận diện bài hát bằng dấu vân tay phổ trên 5 bản thu Wikimedia Commons.

```bash
uv run --python 3.11 --with-requirements prototypes/requirements.txt streamlit run prototypes/streamlit_app.py
```

Streamlit Community Cloud dùng `prototypes/streamlit_app.py` làm entrypoint và Python 3.11.

## Nguồn âm thanh

Các tệp Wikimedia Commons dưới đây được giữ nguyên để ứng dụng khởi động
không cần tải lại. Khi chạy, ứng dụng kiểm tra SHA-256 và dùng 45 giây đầu.

- Für Elise · V Gao (Gaodifan) · CC0. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:FurElise.ogg)
- Jingle Bells · Kevin MacLeod (incompetech.com) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jingle_Bells.ogg)
- Maple Leaf Rag · United States Marine Band. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:1906_-_Scott_Joplin%27s_Maple_Leaf_Rag_%281899%29_played_by_the_United_States_Marine_Band.ogg)
- In the Hall of the Mountain King · Czech National Symphony Orchestra · Musopen. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Musopen_-_In_the_Hall_Of_The_Mountain_King.ogg)
- Giao hưởng số 5 · Skidmore College Orchestra · Musopen. [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ludwig_van_Beethoven_-_symphony_no._5_in_c_minor%2C_op._67_-_i._allegro_con_brio.ogg)
