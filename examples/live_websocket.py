"""A live voice endpoint: the client sends text deltas as its LLM writes them, and gets audio back at once.

    pip install fastapi uvicorn
    uvicorn live_websocket:app

Client protocol: send text messages with pieces of the reply, then the text "<end>". Audio comes back as binary
messages (float32 PCM, mono, 24 kHz). Sending "<stop>" interrupts (barge-in).
"""
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import iterate_in_threadpool

from ema_turbo import EMA

app = FastAPI()
tts = EMA(models="models").warmup()


@app.websocket("/live")
async def live(ws: WebSocket):
    await ws.accept()
    reply = tts.live(sample_rate=24000)

    async def send_audio():
        async for chunk in iterate_in_threadpool(iter(reply)):
            await ws.send_bytes(chunk.tobytes())

    sender = asyncio.create_task(send_audio())
    try:
        while (message := await ws.receive_text()) not in ("<end>", "<stop>"):
            reply.feed(message)
        reply.cancel() if message == "<stop>" else reply.end()
        await sender
        await ws.close()
    except WebSocketDisconnect:
        reply.cancel()
    finally:
        sender.cancel()
