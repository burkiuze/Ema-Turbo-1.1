# ⚡ EMA Turbo

**EMA Lightning'in Türkçe sesi, PyTorch olmadan.** Aynı model, aynı tohumla aynı ses; iki kat hızlı,
beşte bir RAM, yüzlerce MB yerine ~66 MB'lık bir çalışma zamanı. Canlı konuşma için yapıldı: LLM cevabı
yazarken konuşmaya başlar, ilk ses ~50–75 ms'de gelir.

[EMA Lightning](https://github.com/canberk7/ema-lightning) (Canberk Aslan, Apache-2.0) üzerine kurulu:
8.6M parametrelik model ve ağırlıklar onun. EMA Turbo, modeli üç ONNX dosyasına çevirir ve ONNX Runtime +
NumPy ile çalıştırır.

```python
from ema_turbo import EMA            # EMA Lightning'deki EMA ile aynı kullanım

tts = EMA()
speech = tts.say("Merhaba, size nasıl yardımcı olabilirim?", path="merhaba.wav")
```

## Ölçümler

Gerçek ağırlıklar, 2 çekirdekli bulut CPU (Intel Xeon, AVX-512), aynı oturumda ayrı süreçler.

| | EMA Lightning | **EMA Turbo** | EMA Turbo, `low_memory=True` |
|---|---|---|---|
| 60 sn'lik metin | 6.3× gerçek zaman | **13.0×** | 11.9× |
| 10 sn'lik metin | 6.3× | **12.0×** | 11.6× |
| İlk ses (`stream`, aynı metin) | 1208 ms | **141 ms** | — |
| İlk ses, canlı mod (LLM'in ilk kelimesinden) | — | **69–76 ms** | — |
| İlk çağrı, yeni bir makinede | 29.9 sn | **0.21 sn** | 0.28 sn |
| İlk çağrı, toplu boyut ölçülmüşken | 592 ms | **241 ms** | 229 ms |
| RAM, yüklendikten sonra | 548 MB | **114 MB** | 113 MB |
| Tepe RAM | 836 MB | 398 MB | **199 MB** |
| Kurulum | PyTorch + NumPy (PyTorch'un CPU sürümü tek başına yüzlerce MB) | **158 MB toplam** (ONNX Runtime 67 MB, NumPy dahil) | aynı |

Aynı makinede tekrar edilen ölçümlerde EMA Lightning 6.3–8.1×, EMA Turbo 12.0–14.2× gördü; oran her
seferinde yaklaşık 2 kat. "Yeni bir makine" Streamlit Cloud'un her yeniden başlatması gibi önbelleği boş
bir ortam demek. `low_memory=True` sesi değiştirmez: en fazla 5 saniye birden çözer ve belleği
her çağrıdan sonra geri verir.

## Canlı konuşma

Sesli asistanda kullanıcının beklediği tek şey ilk sesin gelmesidir; gerisi çalarken üretilir. EMA Turbo
buna göre ayarlı:

- **Büyüyen pencereler.** Akışın ilk parçası 0.2 saniyelik bir pencere, sonra 0.8 sn, 2 sn ve tam
  pencereler. İlk ses için yalnızca 0.2 saniye çözülür; her sonraki pencere, öncekisi çalıp bitmeden çok
  önce hazır olur.
- **Akışın ilk cümlesi sıra beklemez.** Tek başına düşünülür ve tek başına çözülür; aynı turdaki başka
  cümleler araya girmez. Sıra yine bozulmaz.
- **Tek iş parçacığı havuzu.** Üç model aynı ONNX Runtime havuzunu paylaşır. Ayrı havuzlarda her modelin
  boşta dönen iş parçacıkları bir sonrakinin çekirdeklerini meşgul ediyordu. Bu tek değişiklik ilk sesi
  104 ms'den 61 ms'ye indirdi.
- **Canlı mod.** `live()` ve `stream_text()` LLM'den gelen metni gelir gelmez keser: ilk virgülde ya da
  cümle sonunda. Her parça tamamlandığı an üretilmeye başlar. "Dr.", "3.", "a." gibi noktalar cümle sonu
  sayılmaz. Bir kelime ya da sayı parça parça gelse bile ("14:" "30'da") asla bölünmez. Metin nasıl
  parçalanıp gelirse gelsin kesimler aynı olur, yani aynı tohumla aynı ses çıkar.
- **Söz kesme.** `cancel()` ya da döngüden çıkmak kalan işi anında bırakır. Başka bir iş parçacığı o an
  ses bekliyor olsa bile beklemesi hemen biter.

```python
# LLM'in cevabı yazılırken konuş
for chunk in tts.stream_text(llm_token_stream, sample_rate=24000):
    speaker.write(chunk)

# ya da kendin besle (bir iş parçacığında besle, başka birinde dinle)
reply = tts.live(sample_rate=24000)
reply.feed("Merhaba, ")
reply.feed("size nasıl yardımcı olabilirim?")
reply.end()          # kullanıcı araya girerse: reply.cancel()
for chunk in reply:
    speaker.write(chunk)
```

`examples/live_websocket.py`, bunu bir WebSocket uç noktası olarak kurar: istemci metni parça parça
gönderir, sesi anında geri alır, `<stop>` ile sözü keser. Test istemcisinde ilk ses 67 ms'de geldi.

**Ölçüm**: LLM benzetimi, 44 belirteçlik bir cevap (13 sn konuşma), 2 çekirdekli CPU.

| LLM hızı | Cevabın tamamını bekleyip okumak | Canlı mod | Parça hazır → ilk ses | En dar çalma payı |
|---|---|---|---|---|
| 30 belirteç/sn | 1574 ms | **76 ms** | 42 ms | 159 ms |
| 60 belirteç/sn | 865 ms | **69 ms** | 52 ms | 151 ms |

"En dar çalma payı": her ses parçası, çalınması gereken andan en az bu kadar önce hazırdı. Takılma yok.

`stream()` ile ilk ses (aynı metin, aynı tohum, 5 denemenin ortancası):

| Metin | EMA Lightning | EMA Turbo 1.0 | EMA Turbo 1.1 |
|---|---|---|---|
| Selamlama (2.1 sn) | 548 ms | 164 ms | **66 ms** |
| Hava durumu (4.7 sn) | 1311 ms | 225 ms | **93 ms** |
| Ünlem + soru (3.8 sn) | 971 ms | 183 ms | **78 ms** |
| Sayılar (7.8 sn) | 1631 ms | 335 ms | **115 ms** |
| Telefon, 16 kHz (5.3 sn) | 1398 ms | 194 ms | **94 ms** |
| Haber paragrafı (18.5 sn) | 1057 ms | 364 ms | **127 ms** |

`stream()` sesi `say()` ile aynıdır. Yalnızca çözme pencereleri farklıdır, fark kayan nokta düzeyinde
kalır (pencere kenarlarında 86 dB). Bir metnin tamamı tek parçaysa (EMA Lightning'in bölücüsü 10 saniyeye
kadar metni tek parça tutar), ilk ses için o parçanın tamamı düşünülür. Canlı mod ilk cümleyi ayrı kestiği
için daha erken başlar.

## Aynı ses mi?

Evet. 5 metin × 3 tohum × 2 örnekleme hızı, 30 kliplik karşılaştırmada:

- Her klibin **uzunluğu örneği örneğine aynı**, kelime zamanları aynı.
- Fark, sinyalin **68–103 dB altında** (medyan 81 dB). 16-bit bir WAV dosyasının kendi kuantalama gürültüsü
  ~96 dB'dir; bu fark duyulamaz, yalnızca kayan nokta yuvarlamasıdır.

Neden aynı:

- Modelin matematiği değişmedi. ONNX'e çevrilen grafik PyTorch modeliyle karşılaştırılarak yazılır.
  Dışa aktarma sırasında iki şey değiştirildi; ikisi de gerçek her konumda birebir aynı sonucu verir:
  dikkat maskesi çağrı başına bir kez toplamsal olarak kurulur (`exp(-1e9)` float32'de tam sıfırdır) ve
  `expm1`, `exp - 1` olur (ONNX'te `expm1` yok).
- Tohumdan gelen gürültü aynı. PyTorch'un CPU'daki `randn`'ı (Mersenne Twister + 16'lık Box-Muller
  blokları) NumPy'da birebir taklit edildi: aynı tohum, aynı çekilişler, 1e-6 içinde aynı sayılar.
- Süreleri kareye yuvarlama, kare zaman çizelgesi, yeniden örnekleme ve sıralayıcı (Playhead) EMA
  Lightning'deki adımların aynısı.

`tests/test_equivalence.py`, EMA Lightning kuruluysa iki motoru aynı ağırlıklarla yan yana çalıştırır ve
her tohumda aynı sesi ister.

## Neden daha hızlı

| Değişiklik | Kazanç |
|---|---|
| Dekoder ONNX Runtime'da (CPU'daki sürenin %93'ü dekoderdeydi) | 1 sn ses için 115 ms → 50 ms |
| Dikkat maskesi çağrı başına bir kez (24 dikkatte 48 maskeleme + NaN kontrolü kalktı) | akustik model %24 daha hızlı |
| ONNX Runtime transformer birleştirmeleri (layer norm, GELU) | akustik model ~1.4× |
| Denormaller sıfıra (`set_denormal_as_zero`) | dekoder ~%8 |
| Cümle boyu pencereler (10 sn), kenar payı ödenmiyor | dekoderde %16'ya kadar az iş |
| Akışın ilk penceresi 0.2 sn, sonra büyür; ilk cümle tek başına düşünülür ve çözülür | ilk ses 271 → 141 ms |
| Üç model tek ONNX Runtime iş parçacığı havuzunda | ilk ses 104 → 61 ms, işlem hacmi +%5 |
| İlk çağrı toplu boyut ölçümünü beklemez (EMA Lightning yeni bir makinede ilk çağrıda 1, 2, 4 ve 8'lik toplulukları ölçer) | yeni makinede ilk çağrı 29.9 sn → 0.21 sn |

Denenip elenenler: int8 kuantizasyon (SNR 12 dB, ses bozuldu ve hızlanmadı), Conv2d dönüşümü (yavaş),
ONNX Runtime'ın paralel çalıştırma modu (yavaş).

## Kurulum

```sh
python -m pip install -e .
```

Bağımlılıklar: `numpy`, `onnxruntime`, `normalizer-tr`. PyTorch yok.

Modeller `models/` klasöründe (text 4.8 MB, sound 18 MB, decoder 12 MB, `meta.json`). EMA Turbo modeli
şu sırayla arar: `EMA(models="klasör")`, `EMA_TURBO_MODELS` ortam değişkeni, `~/.cache/ema_turbo`. Modelleri
EMA Lightning'in ağırlıklarından yeniden üretmek için (yalnızca bu adım PyTorch ister):

```sh
python -m pip install -e ".[export]"
python -m ema_turbo.export                    # ağırlıkları Hugging Face'ten indirir
python -m ema_turbo.export --weights klasör   # ya da elindeki ema.pt ve decoder.pt'den
```

## Kullanım

EMA Lightning'in API'si birebir: `say()`, `stream()`, listeler, `Speech`, `Word`, aynı seçenekler.

```python
from ema_turbo import EMA

tts = EMA(models="models").warmup()   # warmup: ilk çağrıyı da hızlı yapar, ölçüp yazar

speech = tts.say("Toplantı 14:30'da başlıyor.", seed=7, sample_rate=24000, path="toplanti.wav")
for w in speech.words:
    print(f"{w.start:5.2f} {w.end:5.2f} {w.text}")

for chunk in tts.stream("Ses, siz dinlerken üretiliyor."):
    speaker.write(chunk)                        # float32, mono

tts.say(["Günaydın.", "Siparişiniz yola çıktı."], path="klipler")
```

| `EMA(...)` | |
|---|---|
| `models` | Model klasörü |
| `threads` | ONNX Runtime iş parçacığı sayısı (varsayılan: fiziksel çekirdek sayısı) |
| `low_memory` | Tepe RAM'i yarıya indirir, ~%10 yavaş, ses aynı |
| `providers` | ONNX Runtime sağlayıcıları (varsayılan CPU) |

| `say()` / `stream()` | Değerler | Varsayılan |
|---|---|---|
| `speed` | 0.25 – 4 | 1.0 |
| `seed` | negatif olmayan tam sayı | rastgele |
| `sample_rate` | 48000, 24000, 16000, 8000 | 48000 |
| `path` (yalnızca `say`) | `.wav` ya da liste için klasör | yok |

`lightning()` da çalışır (EMA Lightning koduyla uyum için `warmup()`'ın diğer adı).

## Streamlit demosu

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

Telefondan Streamlit Cloud'a yüklemek için:

1. GitHub'da yeni bir depo aç, bu klasörü `models/` dahil yükle. En büyük dosya 18 MB; GitHub'ın web
   yükleme sınırı 25 MB, telefondan yüklenebilir.
2. [share.streamlit.io](https://share.streamlit.io) → **Create app** → depo, ana dosya `app.py`, Python 3.12.
3. **Deploy**. PyTorch inmediği için kurulum hızlıdır.

Arayüz Türkçe: metin, hız, örnekleme hızı, sabit tohum, WAV indirme; ilk ses süresi, toplam süre,
gerçek zaman katı ve tepe RAM.

## Sınırlamalar

- **Yalnızca CPU'da ölçüldü.** GPU'da EMA Lightning'in kendi `lightning()` yolu (CUDA grafikleri, RTX
  4090'da 440× gerçek zaman) çok daha hızlıdır; GPU'n varsa onu kullan. `providers` ile ONNX Runtime'ın
  GPU sağlayıcıları verilebilir ama denenmedi.
- **Modelleri üretmek bir kez PyTorch ister.** Çalıştırmak istemez.
- **Tek ses**, ses klonlama ve duygu kontrolü yok: EMA Lightning ne yapıyorsa o. Duygu için Fin Voice
  Turbo var.

## Testler

```sh
python -m pip install -e ".[test]"
python -m pytest -q
```

Testler, EMA Lightning'in mimarisinde küçük rastgele modellerle (`tests/fixtures`, 1.5 MB) çalışır;
PyTorch gerekmez. PyTorch ve EMA Lightning kuruluysa `test_equivalence.py` iki motoru aynı ağırlıklarla
karşılaştırır, `test_rng.py` gürültüyü PyTorch'unkiyle karşılaştırır.

## Lisans ve teşekkür

[Apache-2.0](LICENSE). Değişiklikler [NOTICE](NOTICE) dosyasında.

- **EMA Lightning:** [Canberk Aslan](https://github.com/canberk7/ema-lightning). Model, ağırlıklar,
  Playhead zamanlayıcısı ve metin hattı onun eseri.
- **normalizer-tr:** [Erdem Tuna](https://github.com/erdemtuna/normalizer-tr).

EMA Turbo, Vertex Corporation tarafından geliştirildi.
