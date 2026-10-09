"""EMA Turbo: Streamlit demosu. PyTorch'suz, yalnızca ONNX Runtime.

    streamlit run app.py

Streamlit Cloud'da: depoyu bağla, ana dosya app.py. Modeller depodaki models/ klasöründen okunur (~35 MB).
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

from ema_turbo import EMA  # noqa: E402
from ema_turbo.audio import wav_bytes  # noqa: E402

st.set_page_config(page_title="EMA Turbo", page_icon="⚡", layout="centered")
st.markdown(
    """
    <style>
      .block-container {max-width: 720px; padding-top: 2.2rem;}
      .t-title {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;}
      .t-sub {opacity: 0.7; margin: 0.2rem 0 1.1rem;}
    </style>
    <div class="t-title">⚡ EMA Turbo</div>
    <div class="t-sub">Türkçe metin okuma. EMA Lightning'in sesi, PyTorch'suz: daha hızlı, daha az RAM.</div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Model yükleniyor…")
def load():
    return EMA(models=HERE / "models").warmup()  # ilk tıklama da hızlı olsun


text = st.text_area("Metin", "Merhaba! Siparişiniz yola çıktı, yarın 14:30'da elinizde olacak. "
                    "Başka bir konuda yardımcı olabilir miyim?", height=140, max_chars=5000)
c1, c2 = st.columns(2)
speed = c1.slider("Hız", 0.5, 2.0, 1.0, 0.05)
rate = c2.selectbox("Örnekleme hızı", [48000, 24000, 16000, 8000], format_func=lambda r: f"{r // 1000} kHz")
with st.expander("Gelişmiş"):
    fixed = st.checkbox("Sabit tohum (aynı metin, aynı ses)", value=True)
    seed = int(st.number_input("Tohum", 0, 2**31 - 1, 0, disabled=not fixed))

if st.button("🔊 Seslendir", type="primary", width="stretch"):
    if not text.strip():
        st.warning("Önce bir metin yaz.")
        st.stop()
    tts = load()
    start, first, chunks = time.perf_counter(), None, []
    for chunk in tts.stream(text, speed=speed, seed=seed if fixed else None, sample_rate=rate):
        if first is None:
            first = time.perf_counter() - start  # canlı konuşmada dinleyicinin beklediği süre
        chunks.append(chunk)
    took = time.perf_counter() - start
    if not chunks:
        st.warning("Bu metinde okunacak bir şey bulamadım.")
        st.stop()
    audio = np.concatenate(chunks)
    duration = audio.size / rate
    data = wav_bytes(audio, rate)
    st.audio(data, format="audio/wav")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("İlk ses", f"{first * 1000:.0f} ms")
    m2.metric("Ses", f"{duration:.1f} sn")
    m3.metric("Toplam", f"{took * 1000:.0f} ms", f"{duration / max(took, 1e-6):.1f}× gerçek zaman", delta_color="off")
    m4.metric("Tepe RAM", f"{peak:.0f} MB")
    st.download_button("⬇️ WAV indir", data, file_name="ema-turbo.wav", mime="audio/wav", width="stretch")
    st.caption("İlk ses: sesin ilk parçasının hazır olduğu an. Canlı konuşmada dinleyici yalnızca bunu bekler; "
               "geri kalanı çalarken üretilir.")
