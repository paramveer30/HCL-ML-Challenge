# Web UI

A small local page for the robot, in a plain terminal style: the conversation as a log, an "under the hood" box with the emotion and the numbers behind it, and a box that shows what the robot is looking at. Plain HTML, CSS, and JavaScript, served by FastAPI. No build step.

Nothing leaves the machine. The page loads no fonts, scripts, or styles from anywhere else, and the server only listens on `127.0.0.1`. Nothing on the page is made up: both modes run the real models.

## Run

From the repository root, with the environment active:

```powershell
python -m uvicorn ui.server:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 in a browser. The models load when the page opens (about half a minute the first time), and the buttons stay off until the status says ready. This needs everything the terminal scripts need: the trained checkpoints, the downloaded models, and Ollama running. If something is missing, the conversation box says what.

## Two modes

The switch on the left chooses one, and only that mode's control is shown.

**`[ live ]`** is what the page opens in. You talk to the robot.

- Hold `[ hold to talk ]` or the space bar, say one sentence, and let go. The audio between press and release is the turn. The browser asks for the microphone on the first press.
- The camera switches on by itself (the browser asks once). The box on the right shows the preview with a box around your face and the detector's five points (the eyes, the tip of the nose, the corners of the mouth), and what the face shows, for example `face: joy 0.93`. `camera: [ off ]` turns it off; talking still works with words and voice only.
- Talking again while the robot is still answering cuts it off and starts the new turn.

**`[ sample clips ]`** plays a recorded MELD clip to the robot instead of you.

- Pick a clip and press `[ play clip ]`. Its video plays in the box on the right, its sound comes out of the speakers, and the same models produce the conversation and the numbers.
- Hold to talk is off in this mode and the camera is off.
- A file name starts with the true emotion of each spoken line: `fear_dia40_utt8.wav` is one fear line, `neutral_joy_surprise_dia3.wav` is three lines of one dialogue with 1.5 s of silence between them. After the clip the log adds `sample label: ... the robot said: ...`, so a miss is shown as a miss.
- Each clip starts a new conversation. Inside a dialogue sample the earlier lines are the context for the later ones, which is what the text model was trained on.
- **Expect misses.** On dev the offline model gets 86% of neutral lines right but only 23% of fear and 9% of disgust, and a single line played alone does worse than that, because the robot hears the words through speech recognition and has no dialogue around them (numbers in `DECISIONS.md`).

Switching modes clears the conversation. `[ clear conversation ]` does the same without switching.

## Building the sample clips

The clips are MELD's data and are not in the repository, so the list starts empty. Build them from your own MELD copy (after `python -m meldrt.data.extract_audio`):

```powershell
python scripts/make_samples.py
```

It writes a WAV and an mp4 per sample into `ui/clips/`. Which dev clips and dialogues it takes is the `samples` section of `configs/default.yaml`. Any other 16 kHz mono WAV you put in `ui/clips/` shows up in the list too, with an mp4 of the same name if you have one.

## Reading the page

- The big number next to the emotion is the fused confidence after calibration: on dev, turns shown at 0.8 to 0.9 were right 82% of the time (see `REPORT.md`).
- The table has one column per model: words, voice, face, and the fused result. A turn uses the faces seen while it was spoken; the line under the right-hand box says how many. With no face the turn is words and voice only.
- For your camera the face view comes from a facial expression classifier trained on in-the-wild colour photos (HSEmotion). For the pictures of a MELD clip the head we trained on MELD is used. `vision.live_head` in `configs/default.yaml` switches this. On MELD the face is weak and the fusion weights chosen on dev give it no say in the fused label (see `REPORT.md`), so it shows up as its own column and in the robot's reaction, not in the big label.
- When a confident voice or face disagrees with confident words, the log says so and the robot is told.
- The timing columns: `transcript` is speech recognition, `emotion` is the time added after the transcript until the state is ready, `first word` is from the state to the first word of the reply, and `total` is from the end of the turn to the first spoken audio. For a clip the end of the turn is when the endpointer decides it is over, so the 600 ms of silence it waits for is not in the total.

## How the page talks to the pipeline

`ui/pipeline.py` drives the real pipeline (`build_pipeline`, `push_frame`, `push_image`, `process_turn`, `reply`, the speaker) and turns what happens into events. The events, as JSON over `/ws`:

```
{"type": "status", "state": "idle|loading|unavailable|listening|hearing|thinking|speaking"}
{"type": "turn_start", "turn_id": 1}
{"type": "transcript", "turn_id": 1, "text": "..."}
{"type": "emotion", "turn_id": 1, "fused": {...}, "text": {...}, "audio": {...}, "vision": {...}, "top": "joy", "confidence": 0.72,
 "disagree": false, "face_disagree": false, "faces": 6}
{"type": "reply_delta", "turn_id": 1, "text": "..."}
{"type": "reply_done", "turn_id": 1}
{"type": "timing", "turn_id": 1, "stage": "transcript|emotion|first_word|first_audio", "ms": 420}
{"type": "face", "found": true, "box": [x, y, w, h], "points": [[x, y], ...], "label": "joy", "confidence": 0.93}
{"type": "notice", "text": "..."}
```

The label order is always neutral, joy, surprise, anger, sadness, disgust, fear. Positions in a `face` event are shares of the picture, from 0 to 1. `notice` is a plain message for the conversation log.

The browser sends `{"type": "talk", "on": true}` and `{"type": "talk", "on": false}` around a turn, `{"type": "replay", "clip": "name"}`, `{"type": "clear"}`, `{"type": "frame", "jpeg": "<base64>"}` for one camera image, and the microphone audio as binary frames of 512 samples (16 kHz, mono, 16-bit).

## The text at the top

The title, the by-line, and the email at the top of the page are `TITLE`, `ORG`, `AUTHOR`, and `EMAIL` in `ui/static/config.js`. `CAMERA_ON_IN_LIVE` and `CAMERA_FRAME_MS` in the same file set whether live mode switches the camera on by itself and how often an image goes to the face model.

## Changing the look

Black background and white monospace text with wide spacing. The title is at the top, the row of controls is under it, and the rest of the screen is three outlined boxes side by side: the conversation, "under the hood", and the camera or clip. The page itself does not scroll on a desktop screen; a box scrolls inside itself when it is full. On a narrow screen the boxes sit one under the other. Every color, size, and font is a variable in the `:root` block at the top of `ui/static/style.css`, including one bar color per model. Only fonts already on the computer are used.

## Files

```
ui/server.py              FastAPI app: static files, /clips, /clip-video, /ws
ui/pipeline.py            the real pipeline driven by the page, and the event format
ui/clips/                 the sample clips, built by scripts/make_samples.py (not committed)
ui/static/index.html      the page
ui/static/style.css       the look
ui/static/app.js          WebSocket, the two modes, conversation, probability table, timing, microphone
ui/static/mic-worklet.js  microphone to 16 kHz frames of 512 samples
ui/static/camera.js       the camera preview, still images for the face model, the face tracker, clip videos
ui/static/config.js       switches and the text at the top
```
