"""Speak an LLM's reply while it is still being written. Here a fake LLM writes 30 tokens a second."""
import time

from ema_turbo import EMA

REPLY = ("Merhaba, siparişinizi kontrol ettim. Kargonuz bugün sabah yola çıktı ve yarın öğlene kadar "
         "adresinizde olacak. Başka bir konuda yardımcı olabilir miyim?")


def fake_llm(text, tokens_per_second=30):
    for word in text.split(" "):
        time.sleep(1 / tokens_per_second)
        yield word + " "


tts = EMA(models="models").warmup()
start, played = time.perf_counter(), 0
for chunk in tts.stream_text(fake_llm(REPLY), sample_rate=24000):
    if not played:
        print(f"first audio after {1000 * (time.perf_counter() - start):.0f} ms")
    played += chunk.size  # in a real app: speaker.write(chunk), float32 mono 24 kHz, right away
print(f"{played / 24000:.1f} s of speech")
