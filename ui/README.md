# Web UI

A small local page for the robot, in a plain terminal style: a talk button, the conversation as a log, and an "under the hood" box with the emotion and the numbers behind it. Plain HTML, CSS, and JavaScript, served by FastAPI. No build step.

Nothing leaves the machine. The page loads no fonts, scripts, or styles from anywhere else, and the server only listens on `127.0.0.1`.

## Run

From the repository root, with the environment active:

```powershell
python -m uvicorn ui.server:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 in a browser.

## Mock mode

Mock mode is on by default. It fakes a full turn with small delays, so the page can be shown without any model loaded:

- Hold `[ hold to talk ]` (or the space bar) and let go, or pick a sample and press `[ play ]`.
- The status goes listening, hearing you, thinking, speaking, ready.
- A transcript line appears, then the emotion and the table of probabilities for the text model, the audio model and the fused result, a reply that streams in word by word, and the timing row.
- `http://127.0.0.1:8000/?play=its_fine.wav` replays that sample as soon as the page opens.

The sample names in the list are scripted scenarios, not audio files. `its_fine.wav` and `another_meeting.wav` are the ones where the words and the tone disagree.

## Switching to the real pipeline

The page only talks to the `PipelineAdapter` interface in `ui/pipeline_adapter.py`:

- `list_clips()` returns the names shown in the sample list.
- `run_turn(audio_chunks)` receives the microphone audio as chunks of 512 samples (16 kHz, mono, 16-bit) and yields events.
- `run_clip(name)` replays one sample and yields the same events.

`MockPipeline` and `RealPipeline` both follow it, so the page does not know which one it is talking to.

## Real mode

Untick **mock mode**. The server loads the models (about half a minute the first time) and the log says `models ready.` This needs everything the terminal scripts need: the trained checkpoints, the downloaded models, and Ollama running.

- **Talk:** hold `[ hold to talk ]` or the space bar, say one sentence, and let go. The audio between press and release is the turn. The browser asks for the microphone on the first press, so the first press may need repeating.
- **Replay a sample:** put 16 kHz mono WAV files in `ui/clips/` and they appear in the list. A clip is played out loud and pushed through the voice detector and the endpointer in 32 ms frames at real-time pace, the same way `scripts/replay.py` does it. WAV files are not committed, so the folder starts empty.
- **Samples with a known answer:** if a file name starts with emotion names, they are taken as the true labels of its spoken lines, one per line: `fear_dia40_utt8.wav` is one fear line, `neutral_joy_surprise_dia3.wav` is three lines of one dialogue with 1.5 s of silence between them. After the sample the log adds `sample label: ... the robot said: ...`, so a miss is shown as a miss. Each sample starts a new conversation, and inside a dialogue sample the earlier lines are the context for the later ones, which is what the text model was trained on.
- **Expect misses.** A file name is the true label, not a promise. On dev the offline model gets 86% of neutral lines right but only 23% of fear and 9% of disgust, and a single line played alone does worse than that, because the robot hears the words through speech recognition and has no dialogue around them (numbers in `DECISIONS.md`, "Audit").
- The reply is spoken through this computer's speakers, like `scripts/live.py`.
- `[ clear ]` empties the page and makes the robot forget the conversation.

`RealPipeline` only calls the existing pipeline (`build_pipeline`, `push_frame`, `process_turn`, `reply`, the speaker). The one thing it adds is a small wrapper around the classifier that remembers each model's full probabilities, because the pipeline's state keeps only each model's top label and the page shows all seven.

The timing columns: `transcript` is speech recognition, `emotion` is the time added after the transcript until the state is ready, `first word` is from the state to the first word of the reply, and `total` is from the end of the turn to the first spoken audio. For a replayed clip the end of the turn is when the endpointer decides it is over, so the 600 ms of silence it waits for is not in the total.

The events, as JSON over `/ws`:

```
{"type": "status", "state": "idle|listening|hearing|thinking|speaking"}
{"type": "turn_start", "turn_id": 1}
{"type": "transcript", "turn_id": 1, "text": "..."}
{"type": "emotion", "turn_id": 1, "fused": {...}, "text": {...}, "audio": {...}, "vision": {...}, "top": "joy", "confidence": 0.72,
 "disagree": false, "face_disagree": false, "faces": 6}
{"type": "reply_delta", "turn_id": 1, "text": "..."}
{"type": "reply_done", "turn_id": 1}
{"type": "speaking", "on": true}
{"type": "timing", "turn_id": 1, "stage": "transcript|emotion|first_word|first_audio", "ms": 420}
{"type": "notice", "text": "..."}
```

The label order is always neutral, joy, surprise, anger, sadness, disgust, fear. `notice` is a plain message for the conversation log, for example while the models load or when a turn fails.

The browser sends `{"type": "talk", "on": true}` and `{"type": "talk", "on": false}` around a turn, `{"type": "replay", "clip": "name"}`, `{"type": "mode", "mock": true}`, `{"type": "clear"}`, `{"type": "frame", "jpeg": "<base64>"}` for one camera image, and the microphone audio as binary frames.

## Camera and video

The face model is the third input. It is off until **camera** is ticked.

- **Live:** tick **camera** (the browser asks for permission and shows a preview). While the talk button is held, the page sends one small still image every 200 ms to the local server. The server finds the largest face in each, and the turn's face view is the average over those images. The preview itself never leaves the browser, and nothing is sent while the button is up.
- **Samples:** if a sample has an mp4 with the same name next to its WAV in `ui/clips/`, 8 pictures from it go to the face model at the moments they belong, and the page plays the video in the camera box.
- The table under "probabilities" has one column per model: words, voice, face, and the fused result. The line under the camera box says in how many images a face was seen. With no face the turn is words and voice only.
- When a confident face disagrees with confident words, the log says so and the robot is told to check in, the same way as for the tone of voice.

On MELD the face model is weak and the fusion weights chosen on dev give it no say in the fused label (see `REPORT.md`), so the face shows up as its own view and in the check-in, not in the big label.

`CAMERA_ON_AT_START` and `CAMERA_FRAME_MS` in `ui/static/config.js` set whether the camera starts ticked and how often an image is sent.

## The text at the top

The title, the by-line, and the email at the top of the page are `TITLE`, `ORG`, `AUTHOR`, and `EMAIL` in `ui/static/config.js`.

## Changing the look

Black background and white monospace text with wide spacing. The title is at the top, the talk button is in the middle, and under it are two outlined boxes: the conversation on the left and "under the hood" on the right. On a narrow screen the boxes sit one under the other. Every color, size, and font is a variable in the `:root` block at the top of `ui/static/style.css`, including one bar color per model. Only fonts already on the computer are used.

## Files

```
ui/server.py              FastAPI app: static files, /clips, /clip-video, /ws
ui/pipeline_adapter.py    PipelineAdapter, MockPipeline, RealPipeline
ui/clips/                 your own sample WAV files, and mp4 files with the same names, for real mode (not committed)
ui/static/index.html      the page
ui/static/style.css       the look
ui/static/app.js          WebSocket, emotion line, conversation, probability table, timing, microphone
ui/static/mic-worklet.js  microphone to 16 kHz frames of 512 samples
ui/static/camera.js       the camera preview, still images for the face model, sample videos
ui/static/config.js       switches and the text at the top
```
