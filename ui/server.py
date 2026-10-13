"""Local web UI server: static files, a list of sample clips, and one WebSocket.

Run from the repository root:  python -m uvicorn ui.server:app --host 127.0.0.1 --port 8000
"""
import asyncio
import base64
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ui.pipeline_adapter import MockPipeline, RealPipeline

STATIC_DIR = Path(__file__).parent / "static"
CHUNK_BYTES = 1024          # 512 samples of 16-bit audio

app = FastAPI(title="MELD-RT")
mock_pipeline = MockPipeline()
real_pipeline = RealPipeline()


def pick(mock):
    return mock_pipeline if mock else real_pipeline


@app.middleware("http")
async def no_cache(request, call_next):
    # the page is edited by hand; never let the browser show an old copy of a file
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/clips")
def clips(mock: bool = True):
    return {"clips": pick(mock).list_clips()}


@app.get("/clip-video/{name}")
def clip_video(name: str):
    """The video that belongs to a real sample, so the page can show what the face model is looking at."""
    path = real_pipeline.video_of(Path(name).name)
    if path is None:
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="video/mp4")


async def chunks_from(queue):
    """Turns the queue of audio chunks into an async iterator that ends at the None marker."""
    while True:
        chunk = await queue.get()
        if chunk is None:
            return
        yield chunk


async def forward(websocket, events):
    """Sends every event of a turn to the browser. If the turn fails, the page is told why."""
    try:
        async for event in events:
            await websocket.send_json(event)
    except Exception as error:
        # for example: a checkpoint is missing, or Ollama is not running
        await websocket.send_json({"type": "notice", "text": f"that did not work: {error}"})
        await websocket.send_json({"type": "status", "state": "idle"})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "status", "state": "idle"})
    mock = True
    audio = None            # the queue for the turn being recorded, while the talk button is held
    turn = None             # the task that is running the current turn
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                # binary frames are microphone audio, 512 samples each
                if audio is not None and len(message["bytes"]) == CHUNK_BYTES:
                    audio.put_nowait(message["bytes"])
                continue

            control = json.loads(message["text"])
            busy = turn is not None and not turn.done()
            if control["type"] == "mode":
                mock = bool(control["mock"])
                if not mock and not busy:
                    # start loading the real models now, so the first turn does not have to wait for them
                    turn = asyncio.create_task(forward(websocket, real_pipeline.prepare()))
            elif control["type"] == "clear" and not busy:
                pick(mock).reset()
            elif control["type"] == "frame":
                # one camera image; the face model runs in a worker thread so the audio keeps flowing
                if not mock:
                    jpeg = base64.b64decode(control["jpeg"])
                    await asyncio.get_running_loop().run_in_executor(None, real_pipeline.add_frame, jpeg)
            elif control["type"] == "talk" and control["on"] and not busy:
                audio = asyncio.Queue()
                await websocket.send_json({"type": "status", "state": "listening"})
                turn = asyncio.create_task(forward(websocket, pick(mock).run_turn(chunks_from(audio))))
            elif control["type"] == "talk" and not control["on"] and audio is not None:
                audio.put_nowait(None)
                audio = None
            elif control["type"] == "replay" and not busy:
                await websocket.send_json({"type": "status", "state": "listening"})
                turn = asyncio.create_task(forward(websocket, pick(mock).run_clip(control["clip"])))
            elif control["type"] == "replay" or (control["type"] == "talk" and control["on"]):
                # a turn or the model loading is still running; say so instead of ignoring the click
                await websocket.send_json({"type": "notice", "text": "still busy. try again in a moment."})
    except WebSocketDisconnect:
        pass
    finally:
        if turn is not None:
            turn.cancel()


# last, so /clips and /ws are matched first
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
