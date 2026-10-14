"""Local web UI server: static files, the list of sample clips and their videos, and one WebSocket.

Run from the repository root:  python -m uvicorn ui.server:app --host 127.0.0.1 --port 8000
"""
import asyncio
import base64
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ui.pipeline import PagePipeline

STATIC_DIR = Path(__file__).parent / "static"
CHUNK_BYTES = 1024          # 512 samples of 16-bit audio

app = FastAPI(title="MELD-RT")
pipeline = PagePipeline()


@app.middleware("http")
async def no_cache(request, call_next):
    # the page is edited by hand; never let the browser show an old copy of a file
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/clips")
def clips():
    return {"clips": pipeline.list_clips()}


@app.get("/clip-video/{name}")
def clip_video(name: str):
    """The video that belongs to a sample clip, so the page can show what the robot is watching."""
    path = pipeline.video_of(Path(name).name)
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
        # for example: Ollama stopped running in the middle of a conversation
        await websocket.send_json({"type": "notice", "text": f"that did not work: {error}"})
        await websocket.send_json({"type": "status", "state": "idle"})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    audio = None            # the queue for the turn being recorded, while the talk button is held
    turn = None             # the task that is running the current turn
    # the models start loading as soon as a page connects; the page keeps its buttons off until they are ready
    loading = asyncio.create_task(forward(websocket, pipeline.prepare()))

    def start(events):
        """Starts a turn. If one is still running (the robot is mid-answer), it is cut short first:
        the person talking again matters more than the robot finishing its sentence."""
        nonlocal turn
        if turn is not None and not turn.done():
            pipeline.interrupt()
            turn.cancel()
        turn = asyncio.create_task(forward(websocket, events))

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
            if not loading.done():
                continue
            if control["type"] == "frame":
                # one camera image: the face model looks at it in a worker thread and the page gets back
                # where the face is, so it can draw it
                jpeg = base64.b64decode(control["jpeg"])
                face = await asyncio.get_running_loop().run_in_executor(None, pipeline.look, jpeg)
                if face is not None:
                    await websocket.send_json(face)
            elif control["type"] == "clear":
                pipeline.reset()
            elif control["type"] == "talk" and control["on"]:
                audio = asyncio.Queue()
                await websocket.send_json({"type": "status", "state": "listening"})
                start(pipeline.run_turn(chunks_from(audio)))
            elif control["type"] == "talk" and audio is not None:
                audio.put_nowait(None)
                audio = None
            elif control["type"] == "replay":
                await websocket.send_json({"type": "status", "state": "listening"})
                start(pipeline.run_clip(control["clip"]))
    except WebSocketDisconnect:
        pass
    finally:
        for task in (turn, loading):
            if task is not None:
                task.cancel()


# last, so /clips and /ws are matched first
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
