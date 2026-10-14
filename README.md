# MELD-RT

A small local prototype for an emotion-aware character robot. It listens to one spoken turn, tags the speaker's MELD emotion from what they said (text) and how they said it (audio), and streams back a short reply grounded in both, in text and out loud. With a camera or a video it also reads the speaker's face (vision), as the optional three-modality extension.

Track: **Text + Audio**, plus the optional extension that adds **vision**. Everything runs on one PC with a 6 GB GPU. Nothing is sent to a remote service.

- `REPORT.md`: framing, decisions, results, tradeoffs, limitations.
- `DECISIONS.md`: the log of every decision and measurement, in the order they were made.

## What it does

```
mic or WAV  --32 ms frames-->  voice detector + endpointer
                                      | end of turn
                    +-----------------+------------------+
                    v                                    v
              Whisper small                     WavLM + audio head
                    | transcript                         |
                    v                                    |
            RoBERTa text model                           |
                    +------------> late fusion <---------+
                                       ^
camera or video  --pictures-->  face detector  --largest face-->  ViT + vision head
                                       |
                                 state (JSON)  ---------------> printed
                                       |
                                 reply prompt
                                       v
                           Qwen2.5-1.5B (Ollama, local)  ------> streamed text
                                       |
                               Kokoro, per sentence  ----------> spoken
```

One state per turn:

```json
{
  "type": "state", "turn_id": 1, "transcript": "What?",
  "emotion": {"label": "surprise", "confidence": 0.562, "probs": {"neutral": 0.286, "surprise": 0.562, "...": 0.0}},
  "views": {"text": {"label": "surprise", "confidence": 0.671}, "audio": {"label": "surprise", "confidence": 0.473},
            "vision": {"label": "neutral", "confidence": 0.312}},
  "tone_text_conflict": false,
  "face_text_conflict": false,
  "faces_seen": 6,
  "audio_seconds": 0.7,
  "latency_ms": {"endpoint": 608, "asr": 370, "audio": 333, "text": 12, "state": 386}
}
```

## Results in short

Test split of MELD, scored once (`scripts/final_eval.py`), one seed:

| Model | Weighted F1 | Macro F1 |
| --- | --- | --- |
| Majority class (neutral) | 0.313 | 0.093 |
| Audio only | 0.475 | 0.293 |
| Text only | 0.638 | 0.469 |
| Fused (alpha 0.45, chosen on dev) | 0.637 | 0.459 |

Fusion gained 0.020 weighted F1 on dev and nothing on test. `REPORT.md` discusses why.

The vision extension, same test split (read a second time, on Oct 13, only to add these rows; everything was fixed on dev first, and the four rows above came out identical):

| Model | Weighted F1 | Macro F1 |
| --- | --- | --- |
| Vision only (largest face, frozen ViT + head) | 0.356 | 0.185 |
| Text + vision (weight on text 0.45, chosen on dev) | 0.628 | 0.450 |
| Text + audio + vision (weights 0.45 / 0.55 / 0.00, chosen on dev) | 0.637 | 0.459 |

The face model beats the majority baseline but is weak on MELD, and the weights chosen on dev give it no say in the fused label. In the live system the face is a third view in the state, and the robot checks in when a confident face disagrees with the words.

Delay after a turn ends, in ms, over 50 replayed turns on a GTX 1660 Ti (`scripts/bench_latency.py`):

| | p50 | p95 |
| --- | ---: | ---: |
| State ready | 269 | 462 |
| First reply token | 356 | 529 |
| Full reply | 478 | 734 |
| First spoken sound ready | 578 | 784 |

Before these clocks start, the pipeline waits 608 ms of silence to decide the turn is over. These are with the face model on and each clip's video fed in next to its audio; one video picture takes 70 ms (face detection plus the ViT) while the person is still speaking. Peak GPU memory was 5,446 MiB with every model loaded, of which 1,819 MiB belonged to other programs.

Learned parameters on the inference path: 2,088,681,413, which is 34.8% of the 6 billion cap (`scripts/count_params.py`).

## Requirements

- Windows 10 or 11, an NVIDIA GPU with 6 GB of memory (developed on a GTX 1660 Ti), 16 GB of RAM.
- Python 3.11.
- [ffmpeg](https://ffmpeg.org/) on the PATH.
- [Ollama](https://ollama.com/) installed and running.
- A microphone and speakers for the live script, and a camera for the vision part of the web page.

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
python scripts/fetch_models.py
ollama pull qwen2.5:1.5b
pytest
```

`fetch_models.py` downloads the pretrained models once. After that the live scripts run with the network switched off (`HF_HUB_OFFLINE=1`), and Ollama only listens on `127.0.0.1`.

## Reproduce the models

Trained checkpoints are not in the repository, so the models have to be trained once. On the GTX 1660 Ti the whole chain below takes about an hour after the download, half of it the face features.

1. Download `MELD.Raw.tar.gz` (10.9 GB) from the link in the official [declare-lab/MELD](https://github.com/declare-lab/MELD) README. Unpack it, then unpack `train.tar.gz`, `dev.tar.gz`, and `test.tar.gz` inside `MELD.Raw`.
2. Point `data.meld_root`, `data.wav_root`, `data.feature_root`, and `vision.feature_root` in `configs/default.yaml` at your folders.
3. Run, in order:

```powershell
python -m meldrt.data.extract_audio      # mp4 to 16 kHz mono WAV, about 10 minutes
python -m meldrt.data.meld               # prints row counts and rows without audio
python -m meldrt.train.cache_audio       # frozen WavLM features, about 7 minutes
python -m meldrt.train.train_text        # fine-tunes RoBERTa, about 9 minutes
python -m meldrt.train.train_audio       # trains the audio head, under a minute
python -m meldrt.train.tune_fusion       # picks alpha on dev
python -m meldrt.train.cache_vision      # largest face in 8 frames per clip, frozen ViT features, about 25 minutes
python -m meldrt.train.train_vision      # trains the vision head, under a minute
python -m meldrt.train.tune_fusion3      # picks the three-model weights on dev
python scripts/final_eval.py --split dev # checks the scoring script without touching test
python scripts/final_eval.py             # the one test run
```

Every script that can be interrupted skips what is already done when rerun.

## Run it

Replay a clip through the live pipeline at microphone speed. This is "one input travelling through the system to both outputs":

```powershell
python scripts/replay.py D:/data/MELD/wav/dev/dia0_utt1.wav --label surprise
```

It prints, for each turn:

```
You:    He's lost it, he's totally lost it.
Feels:  sadness 53%  (words: sadness 74%, voice: sadness 36%)
Robot:  I can imagine how that must feel. Let's find a way to help him.
Timing: state 523 ms, first word 596 ms, full reply 711 ms  (after 608 ms of silence)
```

`--json` prints the full state record shown above instead of the two short lines. `--label` only prints the true emotion next to the result; the script never reads the MELD label files. `--no-voice` prints the reply without speaking it, `--fast` drops the real-time pacing. `--video D:/data/MELD/MELD.Raw/dev_splits_complete/dia0_utt1.mp4` feeds the clip's pictures to the face model as well, so all three inputs travel through.

Talk to it:

```powershell
python scripts/live.py
```

Wait for `Listening.`, say one sentence, and pause. It prints what it heard, the emotion, and the reply as it streams, and speaks the reply. While it speaks the microphone is ignored; it prints `Listening.` again when it is ready for the next turn. Stop with Ctrl+C.

If you cannot hear it, or it does not hear you, run `python scripts/check_audio.py`. It plays a tone, records six seconds, reports how much of it was heard as speech, and lists the sound devices. The microphone and speakers can be set with `pipeline.input_device` and `voice.output_device` in `configs/default.yaml`.

Or use the web page, which shows the same turn in a browser: the conversation on the left, and each model's probabilities and the timings on the right.

```powershell
python -m uvicorn ui.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The page starts in mock mode, which needs no models. Untick **mock mode** to load the real ones, then hold the talk button (or the space bar) while you speak. Tick **camera** and the face model sees you while you talk: this is the three-input interaction. See `ui/README.md`.

To run without the face models, set `pipeline.use_vision: false` in `configs/default.yaml`.

## Measure it

```powershell
python scripts/bench_latency.py --turns 50   # p50 / p95 per stage, peak GPU and RAM
python scripts/count_params.py               # counts every model, fails over 6 billion
```

## Layout

```
configs/default.yaml      every path, threshold, and model name
src/meldrt/
  labels.py  schema.py  config.py
  data/      MELD loading, text cleanup, context, audio extraction, video frames and face crops, cached features
  models/    text input format, audio head (also used for vision), face features, late fusion
  train/     feature caching, text, audio and vision training, fusion weight tuning
  eval/      metrics, ablation rows, parameter counting
  pipeline/  voice detector, endpointer, speech recognition, classifier, turn loop, reply, voice
scripts/     final_eval, replay, live, bench_latency, count_params, fetch_models
ui/          the local web page: FastAPI server, pipeline adapter, static files
tests/       run on CPU in seconds, with no dataset and no model downloads
```

`models/`, `data/`, and `eval/` import nothing from `pipeline/`. Only `scripts/final_eval.py` reads the test split.
