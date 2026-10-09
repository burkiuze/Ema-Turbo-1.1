"""One EMA Turbo for every connection: stream speech over a WebSocket. pip install fastapi uvicorn"""
from fastapi import FastAPI, WebSocket
from starlette.concurrency import iterate_in_threadpool

from ema_turbo import EMA

app = FastAPI()
tts = EMA(models="models").warmup()


@app.websocket("/speak")
async def speak(ws: WebSocket):
    await ws.accept()
    text = await ws.receive_text()
    async for chunk in iterate_in_threadpool(tts.stream(text)):
        await ws.send_bytes(chunk.tobytes())  # float32 PCM, 48 kHz, mono
    await ws.close()
