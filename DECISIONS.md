# Decisions

## 2026-10-01: Text + Audio track
Context: the brief asks for one primary track, Text + Audio or Text + Vision.
Decision: Text + Audio. MELD video has multiple faces, cuts, and off-screen speakers, so vision needs face detection and active-speaker tracking before any emotion modeling. Audio is one stream per utterance.
Tradeoff: we lose facial cues, which matter for some emotions (disgust, surprise). Vision is listed as left out.

## 2026-10-01: Real-time means turn-based
Context: the brief asks us to define real-time for our interaction.
Decision: a robot conversation is turn-based. Audio streams in as 32 ms frames, a turn ends after 600 ms of silence, and we target a state within 600 ms and a first reply token within 1.5 s (p50) after the turn ends.
Tradeoff: no reactions mid-sentence and no barge-in. The 600 ms hangover adds fixed delay but avoids cutting people off at short pauses.

## 2026-10-12: Everything on the desktop
Context: two machines available, a desktop with a GTX 1660 Ti (6 GB) and a laptop with an Intel Arc A370M (4 GB).
Decision: train, run, and measure on the desktop only. Most ML tooling targets NVIDIA GPUs, and the Arc has less memory.
Tradeoff: the demo needs a mic on the desktop.

## 2026-10-01: fp32 in PyTorch
Context: the 1660 Ti has no tensor cores and the 1660 series is known for fp16 NaN problems.
Decision: train and run PyTorch models in fp32. Quantized formats inside dedicated runtimes (int8 Whisper, 4-bit GGUF) are fine.
Tradeoff: slower training and more VRAM than mixed precision. The models are small enough that this is acceptable.

## 2026-10-01: Frozen WavLM with cached features (pending approval)
Context: 6 GB VRAM and 16 GB RAM rule out fine-tuning a speech encoder alongside everything else.
Decision: run WavLM-base-plus once over every clip, cache the time-averaged output of all 13 layers, and train a small head with learned layer weights on the cache.
Tradeoff: frozen features cap audio accuracy below a fine-tuned encoder, and mean pooling throws away timing within the clip.

## 2026-10-01: Late fusion with one weight tuned on dev
Context: we need to combine text and audio and prove what each adds.
Decision: $p = \alpha p_{\text{text}} + (1 - \alpha) p_{\text{audio}}$ with $\alpha$ chosen on dev.
Tradeoff: can't learn cross-modal interactions (for example "positive words + flat tone"). We expose disagreement separately as `tone_text_conflict`.

## 2026-10-01: Test split read once
Context: tuning on test inflates reported numbers.
Decision: all selection uses dev. Only `scripts/final_eval.py` scores test.
Tradeoff: none worth taking.

## 2026-10-12: Qwen2.5-1.5B served by Ollama
Context: we need a local, streaming, small LLM that fits next to the other models on 6 GB, on Windows.
Decision: Qwen2.5-1.5B-Instruct, 4-bit, served by Ollama on 127.0.0.1. Chosen over the 3B for lower latency, more VRAM headroom, and an Apache 2.0 license (the 3B has a research-only license).
Tradeoff: replies are a bit less nuanced than the 3B's.

## 2026-10-12: The robot speaks with Kokoro-82M on CPU
Context: a character robot should talk, not print text.
Decision: Kokoro-82M (Apache 2.0) on the CPU, fed one finished sentence at a time while the reply streams, so speech starts after the first sentence. The mic is muted while the robot speaks.
Tradeoff: adds CPU load and some delay before speech starts; no barge-in, so the user can't interrupt the robot. Piper is the fallback if Kokoro is slow or hard to install.

## 2026-10-12: Local scripts instead of a server
Context: the brief asks for an interaction-ready system that accepts input over time.
Decision: `live.py` (mic) and `replay.py` (MELD clip) stream 32 ms frames through the same pipeline. No web UI or network server.
Tradeoff: no remote client. A server would wrap the same pipeline without changing it.

## 2026-10-12: Environment and dependencies
Context: we need a reproducible Python setup on Windows with a CUDA build of PyTorch.
Decision: Python 3.11 venv, PyTorch 2.14.1 from the cu126 index, versions pinned in `requirements.txt`. ffmpeg is a system install on PATH. The dataset lives outside the repo (`D:/data/MELD`), with the path in `configs/default.yaml`.
Dependencies and why: `torch` (models), `transformers` (RoBERTa, WavLM), `accelerate` (required by the Hugging Face `Trainer`), `pandas` (MELD CSVs), `numpy` (cached features), `PyYAML` (config), `soundfile` (reading WAVs), `scikit-learn` (reference F1 to test our own metrics against), `pytest` (tests).
Tradeoff: the PyTorch install is a separate command because of the CUDA index. The README has to say so.

## 2026-10-12: Audio extraction
Context: MELD ships mp4 clips; the models need 16 kHz mono audio.
Decision: one ffmpeg call per clip, 6 in parallel, output cut at 15 s (the live turn cap). Each clip is written to a `.part` file and renamed when ffmpeg succeeds, so rerunning skips finished clips and never trusts a half-written one. Failures go to `failed.txt` per split and are skipped.
Tradeoff: clips longer than 15 s lose their tail in training, the same way a live turn would.

## 2026-10-12: Text cleanup
Context: the CSVs in the raw release store apostrophes, ellipses, and long dashes as cp1252 control characters (U+0092, U+0085, U+0097; 414 apostrophes in dev alone). The copy on GitHub has the same text with curly punctuation instead, and either shows up as junk like `â€™` if read with the wrong encoding.
Decision: read the CSVs as UTF-8 and replace a small explicit map of characters with plain ASCII (`'`, `"`, `...`, `--`), then collapse whitespace. Accented letters are left alone.
Tradeoff: an explicit map only fixes what we listed. Anything new would pass through unchanged.

## 2026-10-12: Data counts after extraction
Context: we need to know exactly which utterances have audio before training anything.
Measured: CSV rows are 9,989 / 1,109 / 2,610 (train / dev / test) in 1,038 / 114 / 280 dialogues, matching the paper. Extracted WAVs: 9,988 / 1,112 / 2,615. Two utterances have no audio: train `dia125_utt3` (the mp4 is corrupt, ffmpeg reports invalid data) and dev `dia110_utt7` (no mp4 in the release). Every test row has a WAV. The dev and test folders hold a few clips with no CSV row, and the test folder has 132 extra `final_videos_test*` files that we ignore. Dev extraction took 33 s for 1,112 clips. Total audio is about 11.9 hours (median clip 2.5 s, 37 clips at the 15 s cap). Feature caching: 13,477 files, 258 MB, every file checked for shape, dtype, and finite values; train and test (12,603 clips) took 6 min 41 s, about 32 ms per clip.
Decision: keep both utterances in the text data and treat them as "no audio", the same as clips under 0.3 s.
Tradeoff: fusion needs a text-only fallback for a handful of utterances.

## 2026-10-12: Clips under 0.3 s get no audio features
Context: measured on the 1660 Ti, WavLM-base-plus on a clip of 0.2 s or less allocates 9.9 GB and takes 2.5 to 4 s. From 0.25 s up it peaks at 387 to 557 MiB and takes 35 to 90 ms (15 s clip: 90 ms).
Decision: `audio.min_seconds: 0.3`. Shorter clips are skipped by the caching script and listed in `skipped.txt`.
Measured on the full dataset: 164 / 20 / 54 clips skipped in train / dev / test. Counting CSV rows, and adding the corrupt and the missing clip, 165 / 21 / 53 rows (about 2% of each split) have no audio features.
Decided: these rows stay text-only. Fusion uses the text probabilities alone for them, and the audio-only row of the results table says how it treats them. We do not pad short clips with silence, because that would invent audio evidence the clip doesn't have.
Tradeoff: those utterances have no audio prediction. The live pipeline needs the same guard.

## 2026-10-12: Training scripts never read the test split
Context: the plan had the text training script save logits for train, dev, and test.
Decision: `train_text.py` loads train and dev only and saves dev logits. Test predictions are made inside `scripts/final_eval.py`, so the rule "only one script touches test" holds with no exceptions for labels. (The audio features for test clips were cached from the WAV files, without opening the test CSV.) We also don't save the text model's predictions on its own training data: nothing in the plan may be fit on them.
Tradeoff: `final_eval.py` has to load both models instead of just reading saved files.

## 2026-10-12: Text model input and training settings
Context: RoBERTa-base fine-tune on a 6 GB card in fp32.
Decision: the input is a sentence pair, `context` and `Speaker: utterance`, which RoBERTa's tokenizer joins as `<s> context </s></s> Speaker: utterance </s>`. Max 128 tokens; when a pair is too long the oldest context tokens are dropped first (measured: 2 of 9,989 train pairs and 0 dev pairs exceed 128; the longest utterance alone is 94 tokens). Batch 8 with gradient accumulation 2, lr 2e-5, 3 epochs, linear decay with no warmup, seed 0. The best epoch is chosen by dev weighted F1.
Dependencies: none new. Evaluation metrics come from scikit-learn (`f1_score`, `confusion_matrix`) with all 7 labels passed explicitly, so a class nobody predicts counts as 0 in macro F1 instead of being dropped.
Tradeoff: one seed only.

## 2026-10-12: Class weights 1/sqrt(n)
Context: neutral is 47.2% of train (4,710 of 9,989); fear and disgust are 268 and 271.
Decision: cross-entropy weights $w_c \propto 1/\sqrt{n_c}$, scaled to mean 1, for both models. On the real counts that is 0.40 for neutral and 1.66 for fear, so a mistake on fear costs about 4 times a mistake on neutral. Full inverse frequency would make it about 18 times.
Tradeoff: milder than full inverse frequency, so rare classes still get under-predicted somewhat, but the majority class isn't wrecked.

## 2026-10-12: Text model result on dev
Measured (seed 0, 3 epochs, 503 s on the 1660 Ti): dev weighted F1 / macro F1 per epoch were 0.591 / 0.432, 0.615 / 0.479, 0.612 / 0.486. Epoch 2 is kept (best dev weighted F1): **0.615 weighted, 0.479 macro**. Always predicting neutral scores 0.252 / 0.085 on dev.
Per class on dev: neutral 0.760, joy 0.605, surprise 0.661, anger 0.418, sadness 0.429, disgust 0.200, fear 0.282.

## 2026-10-12: Audio head settings and result on dev
Context: the head trains on cached features, so we can afford many epochs and pick the best on dev.
Decision: 13 learned layer scores (softmax, start equal), then 768 to 256, ReLU, dropout 0.2, 256 to 7: 198,676 parameters. AdamW, lr 1e-3, batch 64, 30 epochs, same class weights as text, features used as cached with no extra normalization. Trained on the 9,824 train rows that have audio; the best epoch is chosen by dev weighted F1 on the 1,088 dev rows that have audio.
Measured (seed 0, 25 s on the GPU; the CPU gives the same predictions): best epoch 17, **0.502 weighted, 0.361 macro** on dev rows with audio. Over all 1,109 dev rows, predicting neutral for the 21 with no audio, 0.500 / 0.358.
Per class on dev (all rows): neutral 0.663, joy 0.336, surprise 0.437, anger 0.483, sadness 0.356, disgust 0.000, fear 0.233. Audio beats text on anger only.
Learned layer weights (layer 0 to 12): 0.048, 0.043, 0.043, 0.047, 0.043, 0.043, 0.064, 0.094, 0.150, 0.182, 0.121, 0.087, 0.036. Layers 8 to 10 carry 45% of the weight; equal weights would be 0.077 each.
Tradeoff: picking the best of 30 noisy epochs on dev makes the dev number a little optimistic. Test will tell.

## 2026-10-12: Fusion weight chosen on dev
Context: late fusion needs one number, the weight on text.
Decision: grid search over 0, 0.05, ..., 1 on dev weighted F1, smaller alpha on ties. Rows with no audio use text alone. Result: **alpha = 0.45**, dev weighted F1 0.6346 (text alone 0.6150, audio alone 0.5000).
Measured curve on dev: 0.505 at alpha 0, 0.586 at 0.30, 0.623 at 0.40, 0.635 at 0.45, 0.627 at 0.50, then between 0.619 and 0.629 up to 0.90, and 0.615 at 1. So the curve is nearly flat from 0.40 to 0.85, and 0.45 is a single high point on it.
Note: alpha below 0.5 does not mean audio is the better model. The audio head's probabilities are less peaked than the text model's, so they need a larger weight to change the answer at all.
Conflict flag: text and audio top labels differ and both top probabilities are at least 0.4 (`fusion.conflict_min_conf`, pending approval; it does not affect any score).

## 2026-10-12: Final result on test (read once)
Context: `scripts/final_eval.py` is the only code that reads the test split. It was checked first with `--split dev`, where it reproduced the saved dev numbers exactly, and then run once on test at 22:19 with everything frozen: text checkpoint (epoch 2), audio head (epoch 17), alpha 0.45. Seed 0, one seed.

| Model | Dev weighted F1 | Dev macro F1 | Test weighted F1 | Test macro F1 |
| --- | --- | --- | --- | --- |
| Majority (neutral) | 0.2523 | 0.0850 | 0.3127 | 0.0928 |
| Text only | 0.6150 | 0.4792 | 0.6384 | 0.4685 |
| Audio only | 0.5000 | 0.3582 | 0.4747 | 0.2927 |
| Fused (alpha 0.45) | 0.6346 | 0.4951 | 0.6369 | 0.4588 |

Test has 2,610 rows; 2,557 have audio. For the 53 without, audio only predicts neutral and fused uses text alone.

Per-class F1 on test:

| Emotion | n | Text only | Audio only | Fused |
| --- | --- | --- | --- | --- |
| neutral | 1256 | 0.793 | 0.664 | 0.793 |
| joy | 402 | 0.602 | 0.347 | 0.605 |
| surprise | 281 | 0.581 | 0.334 | 0.591 |
| anger | 345 | 0.441 | 0.328 | 0.443 |
| sadness | 208 | 0.415 | 0.243 | 0.395 |
| disgust | 68 | 0.237 | 0.057 | 0.182 |
| fear | 50 | 0.211 | 0.077 | 0.203 |

Fused confusion matrix on test (rows true, columns predicted, order neutral, joy, surprise, anger, sadness, disgust, fear):

```
neutral   1064   66   68   17   29    8    4
joy        100  240   40   14    6    2    0
surprise    36   24  203    9    5    1    3
anger       93   39   63  119    6   17    8
sadness     84   16   18   16   65    4    5
disgust     29    3    9   12    4   10    1
fear        23    3    5    5    6    0    8
```

What it says:
- Both models beat the majority baseline by a wide margin. Text is the strong one.
- **Fusion did not help on test.** It gained 0.020 weighted F1 on dev and lost 0.0015 on test (macro: +0.016 on dev, -0.010 on test). The dev gain did not carry over.
- On test, fusion helps a little on joy, surprise, and anger, and hurts sadness, disgust, and fear.
- Audio alone is weaker on test (0.475) than on dev (0.500). Picking the best of 30 epochs on 1,088 dev rows made the dev number optimistic.
- Likely reasons the dev gain vanished: alpha came from a nearly flat, noisy curve on only 1,109 rows; the audio features are frozen and mean-pooled; and MELD audio has laugh tracks and music.
Decision: report the numbers as they are and keep alpha = 0.45 in the live system. Nothing is retuned after seeing test.
Tradeoff: the live system's fused label is no more accurate than text alone on this data. What audio still adds is the tone/text conflict signal, which the reply uses.

## 2026-10-12: Endpointing rules
Context: the live pipeline gets one speech / no-speech flag per 32 ms frame and has to decide where a turn starts and ends.
Decision: a turn starts after 250 ms of speech in a row and ends after 600 ms of silence in a row or at 15 s. In frames: 8 to start (256 ms), 19 to end (608 ms), 468 at most (14.976 s). The start and end counts are rounded up so we never fire early; the cap is rounded down so a turn never exceeds 15 s. All three are config values.
Tradeoff: 600 ms of waiting is added to every reply. Shorter would cut people off at pauses.

## 2026-10-12: Dependencies for the live pipeline
Context: the live path needs voice activity detection, speech recognition, a local reply model, and audio devices.
Decision and why: `silero-vad` 6.2.3 (speech / no-speech per 512-sample frame; 462,594 parameters counted from the loaded model; 2.2 ms per frame on the CPU), `faster-whisper` 1.2.1 (Whisper small.en through CTranslate2, int8 on the GPU), `sounddevice` 0.5.6 (microphone and speakers), and Ollama serving `qwen2.5:1.5b` on 127.0.0.1 (the reply model).
Measured: faster-whisper loads on CUDA on this machine without extra setup. On four dev clips of 1.3 to 4.9 s it took 285 to 711 ms per clip (the first call is the slowest) and the transcripts matched the gold text apart from punctuation and "talkin'" vs "talking".
Tradeoff: faster-whisper and silero-vad bring `ctranslate2` and `onnxruntime` along.

## 2026-10-12: Live turn loop
Context: frames of 512 samples (32 ms) arrive one at a time; each finished turn has to become one state.
Decisions:
- The pipeline checks the sample rate once and the size of every frame, and raises on a mismatch.
- While waiting for a turn we keep the last 448 ms of frames (the 7 speech frames before the one that triggers the start, plus 200 ms of pre-roll), so the first word is not cut. The silence that ended the turn is dropped before the models see the audio.
- Speech recognition and the audio model are started together in two threads; the text model runs once the transcript exists.
- Whisper decodes greedily (`beam_size: 1`) to keep the delay down.
- A turn with an empty transcript produces no state.
- The live speaker is called "Speaker" in the text model's input, and the context is that speaker's last 2 turns. In training the context held real names and other speakers. This is a known mismatch.
- A turn under 0.3 s gets no audio view: the fused answer is text alone and the conflict flag is false.
- Silero's internal memory is reset after every turn.
Measured on six dev clips pushed through the real pipeline with `HF_HUB_OFFLINE=1`: all models load in 4.4 s; after the first turn (1,003 ms, warm-up) the state is ready 328 to 462 ms after the turn ends, of which speech recognition is 285 to 445 ms, the audio branch up to 254 ms running at the same time, and the text model 12 to 17 ms. PyTorch holds 838 MiB of GPU memory with RoBERTa, WavLM, and the head loaded.
Seen in those clips: a quiet "Oh my God," at the start of one clip was missed because Silero scored it as non-speech, and two clips with long pauses were split into two turns each.
Tradeoff: pauses over 600 ms split one sentence into two turns, and very quiet speech at the start of a turn can be lost.

## 2026-10-12: Reply prompt and streaming
Context: the reply has to be short, spoken to the person, and grounded in both the words and the state.
Decision: one prompt string with fixed rules (talk to the person as "you", at most 2 sentences, about 30 words, plain English), the earlier turns as Person / Robot lines, what was just said, the fused label with its confidence, and what the words alone and the voice alone suggest. Two extra instructions: on a tone/text conflict the robot mentions that the voice sounds different from the words and checks in with one question; when the fused confidence is under 0.45 (`reply.low_conf`, pending approval) it responds to the content and checks in without assuming a feeling. The reply is streamed from Ollama's `/api/generate` on 127.0.0.1 with the standard library, so no new dependency: 64 tokens at most, temperature 0.6, seed 0, model kept loaded for 30 minutes.
What we tried: telling the model not to use quotation marks made it use them more, so that sentence was removed; surrounding quotes are stripped from the stored reply instead. Three wordings of the conflict instruction were compared on six made-up cases; the one kept most often produced a line like "You sound angry even though you say it's just another meeting."
Tradeoff: a 1.5B model does not always follow the rules. It sometimes describes the person instead of answering them, or names the wrong feeling. Reply quality is not scored.

## 2026-10-12: Replay and live scripts
Context: the brief asks for one input travelling through the system to both outputs.
Decision: `scripts/replay.py` reads a WAV, adds one second of silence so the turn can end, and pushes 512-sample frames at microphone speed; `scripts/live.py` does the same from the default microphone through a queue. Both set `HF_HUB_OFFLINE=1`, print the state as JSON, then the reply as it streams, then the timings. Both only import `meldrt.pipeline`. Replay treats each file as its own conversation.
Replay never reads a MELD CSV: the true label of a clip can be typed in with `--label` for display. So the rule that only `final_eval.py` reads the test split still holds. The 10-clip check below used dev clips for the same reason (pending approval: the plan said test clips).
Measured: 10 dev clips replayed in real time, 10 turns. Time after the turn ended, median / max: state 557 / 650 ms, first reply token 722 / 803 ms, full reply 885 / 985 ms. Targets were 600, 1,500, and 3,000 ms. The card showed 3.4 GB in use with every model loaded, including the reply model.
Seen in those clips: the fused label matched the true label on 4 of the 10 first turns, and on 7 of 10 in an earlier run where the clips shared one conversation. Ten clips prove nothing either way, but the live path differs from the scored one in three ways: Whisper's words instead of the transcript, no dialogue context for a lone clip, and a generic speaker name. Its accuracy has not been measured. Several clips also contain more speech than their transcript, or less.
Tradeoff: frames that arrive while a reply is streaming wait in a queue and are processed afterwards.

## 2026-10-12: Voice on the GPU, not the CPU (pending approval)
Context: the plan put Kokoro-82M on the CPU so it would not compete for GPU memory.
Measured on this machine, per sentence of 1.5 to 4.5 s of speech: 590 to 1,670 ms on the CPU, 109 to 182 ms on the GPU, with a peak of 637 MiB of GPU memory. Kokoro has 81,810,022 parameters, counted from the loaded model.
Decision: `voice.device: cuda`. With every model loaded and the voice speaking, the card peaked at 5,371 MiB of 6,144 during a replay of four clips, about 2.1 GB of which was Windows and other open programs.
Tradeoff: about 770 MiB of headroom. On a busier desktop it could run out; setting `voice.device: cpu` trades about a second of delay for that memory.
Dependency: `kokoro` 0.9.4 (brings `misaki`, `spacy`, and `espeakng-loader`). It needs spaCy's `en_core_web_sm` model once; after that it loads with `HF_HUB_OFFLINE=1`.

## 2026-10-12: Speaking while the reply is still being written
Context: waiting for the whole reply before speaking adds the full generation time to the delay.
Decision: the streamed text is cut into sentences; a sentence is sent to the voice as soon as the next one starts (`.`, `!` or `?` followed by a space) or the reply ends. Two worker threads, one that synthesizes and one that plays, so the next sentence is prepared while the current one is playing and the text stream never waits for sound. Both scripts take `--no-voice`.
The microphone hears the robot, so `live.py` waits until the voice has finished plus 200 ms, throws away everything recorded meanwhile, and clears the endpointer. The robot therefore cannot be interrupted (no barge-in), and anything the person says while it talks is lost.
Measured: 10 dev clips replayed in real time with the voice on. Time after the turn ended, median / max: state 519 / 672 ms, first reply token 673 / 788 ms, full reply 871 / 1,114 ms, first spoken audio 977 / 2,049 ms. The 2,049 ms was the first turn, before a warm-up sentence was added at start-up; in a later run of four clips the first turn took 1,221 ms and the others 858 to 937 ms. The target for first audio was 2,000 ms.
Tradeoff: a one-sentence reply is only spoken once it is complete, because we cannot know it has ended until then.

## 2026-10-12: Parameter count
Context: the brief caps the complete local inference path at 6 billion learned parameters, counting total parameters of every required component.
Decision: `scripts/count_params.py` builds the real pipeline and counts from what it loads. PyTorch models (Silero, RoBERTa, WavLM, the audio head, Kokoro) are counted from the loaded objects. Whisper is counted by reading the tensor shapes in the CTranslate2 `model.bin` that faster-whisper loads, and the reply model by reading the tensor shapes in the GGUF file that Ollama loads. Two small extras are included because they are learned and on the path: Kokoro's voice embedding and the spaCy model Kokoro uses to read text. The script exits with an error above the cap.
Measured: Silero VAD 462,594; Whisper small.en 241,761,792; RoBERTa-base with head 124,651,015; WavLM-base-plus 94,381,936; audio head 198,676; qwen2.5:1.5b 1,543,714,304; Kokoro-82M 81,810,022; Kokoro voice embedding 130,560; spaCy `en_core_web_sm` 1,570,514. **Total 2,088,681,413, 34.8% of the cap.**
Tradeoff: buffers that are not parameters (for example normalization statistics) are not counted. Settings stored as scalars in the Whisper file are skipped.
Dependency: `psutil` 7.2.2 for the memory readings in the benchmark (it was already installed as a dependency of `accelerate`).

## 2026-10-12: Latency and memory benchmark
Context: the delays and resource use have to be measured, not estimated, over at least 50 turns.
Decision: `scripts/bench_latency.py` pushes dev clips of at least one second through the live pipeline in 32 ms frames at real-time pace until 50 turns have been processed, one conversation per clip. The voice prepares every sentence but stays silent, so `first_audio` is the moment the first sound is ready for the sound card. The reply model is unloaded first so the GPU baseline does not include it.
Measured (50 turns of 0.5 to 7.4 s, median 1.7 s; ms after the turn was judged over; p50 / p95): speech recognition 563 / 696, audio branch 455 / 521, text model 15 / 20, **state 576 / 713**, **first reply token 732 / 916**, **full reply 968 / 1,327**, **first spoken sound 996 / 1,260**. Endpointing adds 608 ms before all of these. Targets were 600, 1,500, 3,000, and 2,000 ms at p50: all met at p50; the state met its target on 58% of turns and the other three on every turn.
Memory: GPU 1,858 MiB before loading (other programs), 5,365 MiB at peak, so 3,507 MiB for the system, of which PyTorch held 1,569 MiB. RAM peak 2,440 MiB for our process plus Ollama. A browser was open during the run.
A mistake found by measuring: the first version read `nvidia-smi` and the process list twice a second from a second thread. That made the pipeline itself slower: state 1,147 / 1,726 ms, and the text model's p95 was 876 ms instead of 20. Memory is now read between turns only. Before: state p50 1,147 ms. After: 576 ms.
Tradeoff: reading memory between turns can miss a short spike inside a turn. PyTorch's own peak counter (1,569 MiB) does not have that gap.

## 2026-10-13: First live microphone session
Context: the first run of `scripts/live.py` with a person speaking, seven turns.
What worked: every turn was detected, transcribed, and answered. State ready 484 to 620 ms after the turn ended, first reply token 609 to 887 ms, first sound handed to the sound card 821 to 1,100 ms. The robot did not answer itself. On the second "It's fine." the audio model said anger (0.47) against neutral from the words (0.86), and the conflict flag fired.
What did not: no sound was heard. The system default output was a monitor's HDMI audio. Because the script ignores the microphone until the voice has finished, the robot then seemed deaf and slow for several seconds after every reply. The replies also started with "Hello!" on every turn and repeated the previous answer.
Decisions:
- `voice.output_device` and `pipeline.input_device` (default: the system devices), and `scripts/check_audio.py`, which plays a tone, records four seconds, and reports the level and how much the voice detector heard as speech.
- `live.py` prints `listening.` when the voice has finished and the microphone is in use again.
- Prompt: earlier turns are shown as `They said: "..."` / `You answered: "..."` instead of `Person:` / `Robot:`, and one line was added: answer what they just said, do not greet unless greeted, do not reuse earlier answers. Compared on a scripted nine-turn conversation: before, 6 of 7 replies began with "Hello!" and one repeated the previous reply word for word; after, none did. A variant that left the robot's earlier answers out of the prompt also stopped the repetition but echoed the person once.
Tradeoff: the reply model still drifts into filler such as "How can I help you today?".

## 2026-10-13: Chat-style prompt and a readable terminal
Context: in the first live session the replies were stiff and repetitive, and the terminal showed a block of JSON per turn with what was said buried inside it.
Decision, prompt: `build_prompt` now returns chat messages for Ollama's `/api/chat`: one system message with the rules, the earlier turns as real user and assistant messages, and the current turn as the last user message with a one-line private note in square brackets after it. The note is "they sound joy" in the normal case; on a conflict it names what the words and the voice each sound like and asks the robot to say so kindly and ask one short question; under 0.45 confidence it says the feeling is unclear and asks the robot to check in without naming one.
Compared on a scripted eleven-turn conversation with the same model and settings: the single-text prompt opened 4 of the first 4 replies with "Hello!" and ran long; the chat form gave one or two short sentences per turn with no repeated greeting ("That must have been tough. How are you feeling right now?"). On conflicts it was acceptable on 3 of 4 turns and once only half followed the note.
Decision, terminal: both scripts print four short lines per turn (You, Feels, Robot, Timing). `--json` prints the full state record instead. Library warnings and progress bars are silenced in the two scripts.
Tradeoff: the normal-case note carries only the fused label, so the model no longer sees the two separate views unless they conflict. The latency numbers in the benchmark entry above were measured with the earlier prompt and have to be measured again.

## 2026-10-13: Conflict flag needs 0.5 from both models
Context: in live use the audio model's reading of tone was often wrong (it scores 0.475 weighted F1 on test, and a webcam microphone and everyday speech are further from its training data than TV audio). At a threshold of 0.4 the conflict flag fired on weak audio guesses, for example anger at 0.47 on a calm "It's fine.".
Decision: `fusion.conflict_min_conf: 0.5`. Both models must give their own top label at least 0.5 before the robot is told that words and voice disagree. This changes when the robot checks in, not any label or score: the test results do not depend on it.
Tradeoff: real mismatches where the audio model is only moderately sure are now missed. The threshold was chosen from a handful of live turns, not tuned on data.

## 2026-10-13: Fresh-clone check
Context: a reviewer starts from a clone, not from our working folder.
What was done: the repository was cloned into a new folder and every README step was followed in a new Python 3.11 environment: PyTorch, `requirements.txt`, `pip install -e .`, the spaCy model, `fetch_models.py`, `ollama pull`, `pytest`, audio extraction, row counts, feature caching, text training, audio head training, alpha tuning, `final_eval.py --split dev`, one replayed clip, and `count_params.py`. The data folders in the config were the existing ones, so extraction and caching found their outputs and skipped. The test split was not scored again.
Result: every step ran. 117 tests passed in the new environment. The clone held only the public files. The whole run took 22 minutes, 10 of them training the text model while another program shared the GPU.
Reproducibility of the models from the same seed: the audio head came out identical (best epoch 17, dev 0.502 / 0.361). The text model matched in epoch 1 (0.5913) and then drifted slightly: best dev 0.6179 / 0.4835 against 0.6150 / 0.4792 in the original run. Alpha was 0.45 again, with fused dev 0.6301 against 0.6346. The parameter count was the same, 2,088,681,413.
Tradeoff: fine-tuning on the GPU is not bit-for-bit repeatable, so a reviewer who retrains should expect dev scores within about 0.005 of ours, not the same digits.

## 2026-10-13: A local web page for the robot (mock mode first)
Context: the terminal scripts show the state and the reply, but a character robot is easier to judge when the emotion is visible. Earlier the plan left a web UI out for time; blocks A to C finished early enough to add one.
Decision: a `ui/` folder with a FastAPI server (static files, `/clips`, one WebSocket at `/ws`) and a page in plain HTML, CSS, and JavaScript with no build step. The page is a plain terminal-style screen (black, white monospace, labels in square brackets): the current emotion as one large line, the conversation as a log, and an "under the hood" panel with the text, audio, and fused probabilities and the timings. A first version had an animated character with a warm palette; it was replaced by this simpler look the same day. The page only talks to a `PipelineAdapter` interface. `MockPipeline` fakes a full turn; `RealPipeline` is a stub, so the existing pipeline code is untouched. Microphone audio is converted in the browser to 16 kHz frames of 512 samples, the size the voice detector expects.
Checked in a headless browser in mock mode: status changes, transcript, three bar groups, streamed reply, timing rows. No console errors. Every request went to 127.0.0.1. No horizontal scrolling at 375 px wide.
Dependencies: `fastapi` 0.143.0, `uvicorn` 0.54.0, `websockets` 17.2. They are only needed for the page, not for the scripts.
Tradeoff: more code to keep working, and the page is not connected to the real models yet. The server binds to 127.0.0.1, so nothing leaves the machine.


## 2026-10-13: The web page runs the real models
Context: the page was mock only. A demo needs the real pipeline behind it.
Decision: `RealPipeline` in `ui/pipeline_adapter.py` calls the existing pipeline and changes nothing in `src/meldrt/`. Models load when mock mode is unticked. A push-to-talk turn is the audio between press and release, sent straight to `process_turn`, so the voice detector and the endpointer are not used for it. A replayed clip from `ui/clips/` goes through `push_frame` in 32 ms frames at real-time pace, like `scripts/replay.py`, and is played out loud. The reply is spoken by the server through the computer's speakers. The pipeline runs in a worker thread with a lock, one turn at a time. A wrapper around the classifier keeps each model's full probabilities for the page, because the state only keeps each model's top label.
Layout: the page was reworked twice after looking at it. Final form: title and by-line centered at the top, the talk button in the middle, two outlined boxes under it (conversation, under the hood), black and white monospace with wide spacing. The three bar groups became one table with a column per model.
Checked: through the WebSocket in real mode, two dev clips and one push-to-talk turn gave a transcript, three distributions, a streamed and spoken reply, and timings (first audio 992 to 1,122 ms after the end of the turn). In a headless browser with a dev clip as a fake microphone, holding the space bar gave a full turn with no console errors. The tests use a fake pipeline and need no models.
Tradeoff: push-to-talk has no endpointing, so the page does not show off the endpointer on live speech; `scripts/live.py` does. The first press loses the first words while the browser sets up the microphone. The page timings are single turns, not the benchmark.

## 2026-10-13: Audit of the weighting choices and of the live path (dev only)
Context: before the write-up we checked that the class weights, the fusion weight, and the criteria behind them hold up, and why a sample labelled fear came out as neutral in the web page. Everything here uses the dev split. The shipped checkpoints, alpha, and the test scores did not change.
Confirmed:
- `scripts/final_eval.py --split dev` reproduces the saved numbers exactly (text 0.6150 / 0.4792, audio 0.5000 / 0.3582, fused 0.6346 / 0.4951), so the checkpoints on disk are the reported ones.
- The class weights are computed from train counts only and match the table (0.40 for neutral to 1.66 for fear, a ratio of 4.2; full inverse frequency gives 17.6).
- Audio head, three seeds per scheme, dev weighted / macro F1: no weights 0.474 to 0.479 / 0.303 to 0.317 (fear F1 0.00 in all three); `sqrt_inv` 0.489 to 0.502 / 0.353 to 0.361; `inv` 0.419 to 0.435 / 0.314 to 0.339. `sqrt_inv` is the best of the three on both scores in every seed.
- Text model, one seed per scheme (two extra training runs written to `runs/audit/`, the shipped checkpoint untouched), best dev weighted / macro F1: no weights 0.608 / 0.456 (fear F1 0.12); `sqrt_inv` 0.615 / 0.479 (fear 0.28); `inv` 0.590 / 0.474 (fear 0.26). `sqrt_inv` is the best of the three on both scores. Its lead over no weights in weighted F1 (0.007) is about the size of the seed-to-seed drift we measured (0.005); the lead in macro F1 (0.023) is larger.
- Fusion on dev: +0.0196 weighted F1 over text. Resampling the 1,109 dev rows 2,000 times gives a 95% interval of +0.002 to +0.037, and fusion is ahead in 98% of resamples. Alpha 0.45 is the best value in 72% of resamples (0.70 in 15%). So the dev gain is real but small, and its lower end is close to zero, which fits the test result (-0.0015).
- The live audio branch equals the offline one: same top label on 248 of 248 clips, largest probability difference 0.0000.
Found:
- The live path loses accuracy, and most of it is the recognized words. First 200 dev clips, weighted / macro F1: offline fused 0.658 / 0.525; gold words, context kept, speaker called `Speaker` 0.650 / 0.581; gold words, no context 0.615 / 0.535; Whisper's words, no context (the live path for a lone clip) 0.523 / 0.379. Whisper small.en with beam 1 has a word error rate of 29% on these clips and writes 87 of the 168 `!` and `?` marks the gold lines have.
- Share of clips of each emotion the live path gets right (248 clips): neutral 0.87, joy 0.50, surprise 0.45, anger 0.32, sadness 0.25, fear 0.21, disgust 0.05. Offline fused on the same clips: 0.83, 0.75, 0.76, 0.45, 0.50, 0.23, 0.09.
- A bug in the web page: a replayed sample kept the previous sample as its dialogue context. It now starts a new conversation, like `scripts/replay.py`.
- A crash: a transcript longer than the text model's 128 tokens with no context to drop made the tokenizer fail. Whisper produced a 120-word transcript for a 0.7 s clip when its retry at higher temperature was switched off. `keep_end` in `pipeline/classify.py` now keeps the end of such a transcript. Training and evaluation are not affected (the longest MELD line is 94 tokens).
Decisions:
- Sample clips in the page whose file names start with emotion names are compared with what the robot said, in the log, so a miss is shown as a miss. Dialogue samples (several lines of one dev dialogue in one file) give the text model real context.
- The sample clips used for the demo (local files, not committed) were picked from dev by a rule: for each emotion, the clip the live path was most sure of among those it got right. It got no disgust clip right, so the disgust sample is a known miss. Replayed through the page: 6 of 7 single lines right, and 2 of 3 and 2 of 4 lines right in two dialogue samples.
- The models stay as they are. Making them better on rare emotions means retraining and scoring test again.
With more time: fine-tune the text model on recognized transcripts, or use a larger recognizer; several seeds; calibrate the two models before fusing.

## 2026-10-13: Faster answers
Context: the delay after a turn was dominated by speech recognition and the audio model running at the same time on one GPU.
Measured first, each change on its own:
- Whisper compute type, 200 dev clips, recognition alone: `int8` 325 ms p50, `int8_float32` 204 ms. Word error rate against the gold lines 0.289 and 0.284; live-path weighted F1 0.523 and 0.516 (a difference of one or two clips). Leaner decoding (no timestamps, no retry at higher temperature) saved another 16 ms but allowed the runaway transcript described above, so decoding is unchanged. Beam 5 cost 115 ms for +0.006 F1 and was not taken.
- Order, 16 dev clips, `int8_float32`: side by side 249 ms for the pair (the audio model alone takes 22 ms, next to Whisper 126 ms); one after the other 224 ms.
- Idle GPU, 14 dev clips: the pair took 207 ms after 1 s of idle and 421 ms after 4 s. A tiny GPU job per frame did not help (418 ms). The audio model on half a second of silence once per frame during the last 0.6 s did (228 ms).
- End-of-turn wait, 400 dev clips: cut into two or more turns 11.5% at 600 ms, 14.2% at 500, 18.0% at 400, 23.8% at 300. 10.8% of clips are not detected as a turn at any setting.
- Whisper on silence, noise, or hum writes "you" with `no_speech_prob` 0.86 to 0.94. On 245 real clips with a transcript the highest value was 0.594.
Decisions:
- `asr.compute_type: int8_float32`.
- `process_turn` runs the audio model, then speech recognition. The thread pool is gone.
- `pipeline.warm_during_silence: true`: while the endpointer is inside a turn and the current frame is silent, the audio model runs on half a second of silence and the result is thrown away.
- `asr.max_no_speech_prob: 0.75`: a turn whose every segment is above it gives an empty transcript, so no state and no reply.
- `pipeline.end_silence_ms` stays 600. The rule was "at most 1 point more cut-offs than at 600"; 500 ms adds 2.7 points.
A check that failed as written: the plan said the new recognition settings must give the same words as before on 98% of clips. Changing the compute type alone changes the words on 7.5% of clips, because greedy decoding flips on small numeric differences. The check was replaced by word error rate against the gold lines and F1 against the gold labels, both unchanged within noise.
Benchmark, 50 turns, same clips and prompt, p50 / p95 in ms after the turn was judged over. Before: recognition 579 / 919, audio 469 / 550, state 596 / 934, first reply token 676 / 1,019, full reply 808 / 1,178, first sound 895 / 1,236. After: recognition 212 / 436, audio 24 / 35, state 253 / 489, first reply token 334 / 564, full reply 450 / 745, first sound 542 / 780. The state is ready within 600 ms on 98% of turns (50% before). GPU peak 4,508 MiB with 1,137 MiB used by other programs, so 3,371 MiB for the system (3,475 before). The numbers in the Oct 12 benchmark entry were measured with the earlier prompt and are replaced by these.
Tradeoff: the GPU does throwaway work during pauses inside speech, up to 19 calls of about 20 ms per pause. Push-to-talk in the web page has no wait for silence, so it does not get this warm-up. The 608 ms wait still comes before every number above.

## 2026-10-13: Vision extension (three modalities)
Context: the brief's optional extension asks for vision, voice, and text in one real-time system. The core track was finished, so vision was added in the evening of Oct 13. This changes the earlier decision to leave vision out.
Decision, features: for every clip, 8 frames spread evenly over the first 15 s. In each frame the largest face found by YuNet (through OpenCV, detection on a copy shrunk to 640 px wide, faces under 40 px ignored) is cropped square with a 25% margin and resized to 224 px. A frozen ViT-base trained for facial expressions (`trpakov/vit-face-expression`) gives the [CLS] token of its 13 layers per face; these are averaged over the clip's faces and cached as 13 x 768 float16, the same shape as the audio features. Caching reads the mp4 files only, never a label file.
Decision, model: the same head as audio (learned layer mix, 768 to 256 to 7, 198,676 parameters), same class weights, same training loop, best epoch by dev weighted F1.
Measured, faces: found in 9,928 of 9,989 train clips, 1,103 of 1,112 dev clips, 2,597 of 2,615 test clips. Rows with no face: 61 / 10 / 18. Median face size on 30 dev clips 129 px. Caching all 13,716 clips took about 25 minutes.
Measured, dev: vision only 0.348 / 0.208 on rows with a face, 0.345 / 0.207 over all rows (majority 0.252 / 0.085), best epoch 17. Per class: neutral 0.53, joy 0.31, anger 0.27, sadness 0.23, surprise 0.11, disgust 0.00, fear 0.00. It predicts neutral for 667 of 1,099 rows. Layer weights peak at the last two ViT layers (0.143, 0.139; equal would be 0.077).
Checked on dev before settling: (a) the ViT's own expression classifier with no MELD training, applied to the cached last-layer features: 0.257 / 0.171, so training the head on MELD is worth 0.09; (b) seven head variants (layer norm, z-scored inputs, lr 3e-4, dropout 0.5, last layer only), two seeds each: 0.344 to 0.361 best-epoch weighted F1, so the head is not the limit and the first one was kept.
Decision, fusion: `tune_fusion3.py` searches every (text, audio, vision) weight triple on a 0.05 grid on dev weighted F1, smaller vision weight on ties. Rows with no face keep the two-model fusion with alpha 0.45, so a system without a camera is unchanged. Result: **(0.45, 0.55, 0.00)**, 0.6346, the same as text + audio. The best triple with any vision is (0.50, 0.20, 0.30) at 0.6332. Best dev score per vision weight: 0.635 at 0, then 0.629 to 0.633 from 0.05 to 0.5. Text + vision with one weight: 0.45 on text, dev 0.6273 (text alone 0.6150).
We kept the weights the rule chose, with vision at 0, and did not pick a triple with vision in it afterwards: the differences are far below the noise we measured for fusion on dev (95% interval about 0.035 wide).
Test, read a second time (Tue Oct 13, 19:41) only to add three rows, everything fixed on dev first: vision only **0.3557 / 0.1853**, text + vision **0.6283 / 0.4496**, text + audio + vision **0.6369 / 0.4588** (equal to the two-model fusion). The four earlier rows came out identical to the Oct 12 run (0.3127, 0.6384, 0.4747, 0.6369). The Oct 12 results file is kept as `runs/final_eval/test_oct12.json`.
Decision, live: `Pipeline.push_image` takes one camera or video picture, finds the face, runs the ViT, and keeps the result with its arrival time. `process_turn` uses the faces that arrived while the turn was being spoken, averages them like the cached features, and runs the head. The state gains `views.vision`, `face_text_conflict` (same rule and threshold as the tone flag), and `faces_seen`. The reply note names a face that disagrees with the words ("your words sound joy, but your face looks sadness"). With the weights above the face does not move the fused label; it is a third view and a reason to check in.
Who is right when the face flag fires on dev (both at least 0.5 sure, 23 of 1,099 rows, 2.1%): the words 48%, the face 22%, neither 30%. For the tone flag at 0.5 (83 rows): words 60%, tone 19%.
Decision, page: a camera switch (off by default). While the talk button is held the browser sends one 320 x 240 JPEG every 200 ms over the WebSocket; nothing else leaves the browser, and the server only listens on 127.0.0.1. A replayed sample with an mp4 next to its WAV feeds 8 pictures in at the moments they belong, and the page shows that video. The probability table has a fourth column for the face. `scripts/replay.py --video` and `scripts/bench_latency.py` do the same from the terminal.
Checked live: six dev samples with video through the page's server in real mode gave 1 to 8 faces per turn and a face view on every turn.
Dependencies: `opencv-python-headless` 5.0.0.93 (video decoding, face detection, JPEG decoding). The detector file (232 KB) and the ViT are fetched by `scripts/fetch_models.py`.
A mistake found on the way: the first caching run handed every clip to the worker processes at once; they found faces faster than the GPU used them, the crops piled up, and the run died with an out-of-memory error at clip 7,500. The script now works in batches of 48 clips, with the workers one batch ahead of the GPU.
Tradeoffs and limits: the largest face is not always the speaker (visible on the contact sheet of crops: in 4 of 16 clips the largest face changes person mid-clip), and MELD labels the utterance, not the facial expression, so the face is a weak signal here. No active-speaker detection. The head was trained on TV frames; the webcam path works but its accuracy was not measured. `pipeline.use_vision: false` runs the system without the face models.
Latency and size with vision on (50 turns, each clip's video fed in next to its audio; p50 / p95 in ms after the turn was judged over): recognition 228 / 415, audio 26 / 37, vision head 1 / 1, state 269 / 462, first reply token 356 / 529, full reply 478 / 734, first sound 578 / 784. One video picture costs 70 / 89 ms (face detection plus the ViT) while the person is still speaking; 49 of the 50 turns had a face. GPU peak 5,446 MiB with 1,819 MiB used by other programs, so 3,627 MiB for the system (3,371 without vision). Parameters: 2,174,731,866, 36.2% of the cap (YuNet 53,121, ViT 85,798,656, vision head 198,676 added).

## 2026-10-13: Calibrated confidence
Context: live turns showed confidences of 0.3 to 0.5 even when the label was plainly right.
Measured on dev: the fused model is right 65.6% of the time, with a mean confidence of 0.511 (calibration error 0.145). At a shown confidence of 0.4 to 0.5 it is right 59% of the time, at 0.5 to 0.6 71%. For comparison: text alone, accuracy 0.630 and mean confidence 0.694; audio alone, 0.514 and 0.444; vision alone, 0.381 and 0.328. The fused number is too low because averaging a sharp distribution with a flat one flattens it.
Decision: one temperature for the fused probabilities, fitted on dev by maximum likelihood on a grid of 0.05 (`train/calibrate.py`, written to `runs/fusion/calibration.json`, applied in `Pipeline.process_turn`). The probabilities are raised to the power 1/T and rescaled, so the order of the labels, and with it every F1 score, stays the same. Result: **T = 0.70**, mean confidence 0.634, calibration error 0.038. After calibration: 0.7 to 0.8 is right 76% of the time, 0.8 to 0.9 82%, above 0.9 91%.
Stability: fitted on dialogues with an even number, T = 0.75, and on the odd half accuracy 0.676 against mean confidence 0.609; fitted on the odd ones, T = 0.70, accuracy 0.636 against 0.635.
`reply.low_conf` moved from 0.45 to 0.5: on the calibrated scale, below 0.5 the label is right about half the time.
Tradeoff: fitted and checked on dev only; test was not read for this. The per-model views stay uncalibrated, and the conflict flags still use them. Confidence cannot go higher than accuracy does: on MELD that is about two in three.

## 2026-10-13: A different face head for a live camera
Context: in front of a webcam the MELD-trained vision head read a smile as surprise. It was trained on TV shots where the largest face is often not the speaker and the label belongs to the utterance.
Decision: `vision.live_head: expression`. For pictures from a live camera the face view is the ViT's own 7-way expression classifier (same checkpoint, 5,383 more parameters), averaged over the pictures of the turn. For pictures from a MELD clip's video the MELD-trained head is used, as before. `Pipeline.push_image(frame, from_camera)` carries the difference.
What we know: on 120 dev clips the expression classifier scores 0.261 weighted F1 (accuracy 0.267, mean confidence 0.57) against 0.350 for our head, so on MELD it is worse, and when it was briefly used for the sample clips it raised false "face disagrees" flags (joy 0.71 on a sad line). That is why clips keep our head.
What we do not know: how good it is on a webcam. There is no labelled webcam data here. In a headless browser with a dev clip as the camera it read a laughing face as joy 0.99.
It cannot change a reported score: the vision weight in the fusion is 0, so the face only appears as its own view and in the check-in.
Parity check after the change: with the MELD head, the live path gave the cached top label on 120 of 120 dev clips, largest probability difference 0.0000. `scripts/final_eval.py --split dev` prints the same seven rows as before.

## 2026-10-13: Replies that react to what was said
Context: the robot kept answering "I understand", "How can I help you?", and once read its private note out loud.
Measured first: a fixed script of 16 turns in five short conversations (good news, a death, a late train, a joke, a question, an exam, two words-against-voice conflicts, one words-against-face conflict, "Hmm."), same seed, counting stock assistant phrases, leaks of the instructions, and replies that reuse a content word the person said.

| Prompt and model | Stock phrase | Instruction leak | Picks up a word they said | First word, p50 |
| --- | ---: | ---: | ---: | ---: |
| old prompt, qwen2.5:1.5b | 2 of 16 | 2 | 6 | 100 ms |
| new prompt, qwen2.5:1.5b, no stop phrases | 6 of 16 | 1 | 14 | 109 ms |
| new prompt, qwen2.5:1.5b, stop phrases (kept) | 1 of 16 | 0 | 12 | 110 ms |
| new prompt, llama3.2:3b | 0 of 16 | 0 | 8 | 158 ms |

Decision, prompt: the rules now say what the robot is (a companion next to the person, not an assistant) and what a good answer contains (one concrete detail from what was said, a reaction that fits, at most one question about it). Four example turns are given as if they had happened. How the person seems moved from a note after their words to one short line in round brackets before them ("They seem happy.", "Their words sound calm, but their voice sounds sad."), and a calm person gets "Nothing stands out in how they sound." so there is nothing to comment on.
Decision, stop phrases: telling the 1.5B model what not to say made it say it more often (6 of 16). Instead `reply.stop` lists the openings of the stock endings ("How can I", "Is there anything", "Let me know", ...) and generation ends where one would begin. The useful first sentence stays.
A larger reply model was tried and not kept: `llama3.2:3b` gave the most natural replies ("Classic joke! Did you tell that one to anyone recently?") and the parameter total would still be 3.84 billion, under the cap. But it needs 2,436 MiB of GPU memory against 1,112, and with everything loaded the card was at 5,730 of 6,144 MiB. The benchmark got worse everywhere: speech recognition 722 / 1,876 ms (241 / 453 with the small model), state 765 / 1,934, first sound 1,154 / 2,266. The rule set beforehand was a first sound under 1 s at p50, so the 1.5B model stays. `reply.model` is one line in the config for a card with more memory.
Example with the kept setup: "The train was late again and I missed my meeting." gets "The train being late is no fun. Did you get a chance to reschedule?" (before: "I understand you're frustrated. Maybe we could look for another time.").
Tradeoff: a reply that begins with a stock phrase comes out empty; this did not happen in the 16 turns. The script is ours and small; reply quality on real conversations is not scored. The model still gives advice sometimes ("You should start studying now").

## 2026-10-13: Talking over the robot, loading as a state, and warm-up for push-to-talk
Context: the page answered "still busy, try again in a moment" when the talk button was pressed while the robot was still speaking, or while the models were loading.
Decision: a new turn interrupts the running one. `Speaker.stop()` drops the queued sentences and cuts the sound; the reply loop stops at the next piece; what was said so far is kept in the history. While the models load the server sends the status `loading` and the page keeps the talk and play buttons off. The notice is gone.
Decision: while the talk button is held, the audio model runs on half a second of silence every 128 ms (`Pipeline.keep_warm`), the same idea as on the microphone path with endpointing. With the camera on, the face model already keeps the GPU busy.
Checked: three push-to-talk turns in a row through the WebSocket, the second cut off by the third as its reply started: no notice, the third and fourth turn normal (first sound 913 and 752 ms). Five presses in a row in a headless browser, with and without the camera, each gave a turn.

## 2026-10-13: Face tracker on the camera, clearer controls
Decision: `FaceFinder.largest_face` also returns the detector's 5 landmarks (eyes, nose tip, mouth corners), which YuNet computes anyway. With the camera on and the real models chosen, the page sends one 480-pixel-wide still about every 150 ms (the next one only after the answer to the last), and the server answers with the box, the points, and what the face shows. The page draws them over the preview. Faces only count for a turn if they arrived while it was spoken, as before.
Decision: the controls are one row. Sample clips on the left, the talk button in the middle, on the right two labelled switches: models (scripted demo or real) and camera (off or on), each with a line saying what it does. "Mock mode" is now called "scripted demo".
Checked in a headless browser at 1920 x 950 with a dev clip as the camera: the box and points are drawn on the face, the line under the preview reads `face: joy 0.99`, 59 images with a face in one turn, no console errors. The page fits one screen at six sizes from 1280 x 650 to 2560 x 1300.

## 2026-10-13: Tidying
- `train_vision.py` was a copy of `train_audio.py`. The loop is now in `train/train_head.py` and both entry points are a few lines. Retraining both heads with it gave files identical to the shipped checkpoints (same weights, same dev logits).
- The head used for audio and vision is `LayerMixHead` in `models/head.py` (it was `AudioHead` in `models/audio.py`). The saved weights load unchanged.
- `ui/pipeline_adapter.py` became `ui/adapter.py`, `ui/mock_pipeline.py`, and `ui/real_pipeline.py`. The wrapper that remembered each model's probabilities for the page is gone: `View` now carries them.
- Removed: `face_features`, the `speaking` events the page ignored, the reserved `SOUND_CUES` switch, an unused return value in `cache_vision.py`.
Latency and size after all of today's changes (50 turns with each clip's video, p50 / p95 in ms after the turn was judged over): recognition 241 / 453, audio 28 / 37, vision head 1 / 1, state 279 / 516, first reply token 352 / 635, full reply 497 / 834, first sound 582 / 844; one video picture 67 / 94 ms. GPU peak 4,788 MiB with 975 MiB used by other programs, so 3,813 MiB for the system. Parameters 2,174,737,249, 36.2% of the cap.

## 2026-10-13: Two modes on the page, both real; the made-up demo is gone
Context: the page had two switches that could be combined four ways: made-up or real models, and talk or play a sample. In the made-up mode holding talk played an invented turn and pressing play showed no video, because that mode had no clips behind it at all. It was confusing, and an invented result next to a real clip would have looked like a real one.
Decision: one switch with two modes, and only the current mode's control is shown.
- `[ live ]`, where the page opens: the camera switches on, hold to talk works, the face tracker is drawn on the preview.
- `[ sample clips ]`: pick a MELD clip and play it. Its video shows where the camera was, its sound goes through the same pipeline at microphone pace, and the log ends with the true label against what the robot said. Hold to talk and the camera are off.
Switching modes clears the conversation, so a clip is never context for what the person says.
The made-up pipeline (`MockPipeline` and its scripted scenarios) is deleted. With one pipeline left, the interface class had no purpose either: `ui/pipeline.py` now holds `PagePipeline`, which replaces the three modules from earlier today. The models load when a page connects; if they cannot be loaded the status says so and the conversation box names what is missing.
Decision, samples: `scripts/make_samples.py` builds the sample clips from the dev split (the list is `samples` in the config): single lines are copied, and the lines of a dialogue are joined with 1.5 s between them, the video with ffmpeg as H.264. The first joined videos used a codec browsers do not play, which is why dialogue samples showed no picture. The clips are MELD's data and stay out of the repository.
Checked in a headless browser at 1920 x 950 with a dev clip as camera and microphone: the page opens in live with the buttons off while loading, then the camera on and the face tracked; a held space bar gives a turn with 29 faces; in sample mode the talk button is gone, the space bar does nothing, the camera is off; a single line and a three-line dialogue both play their video (the player's time advances, no error) and end with the label line; back in live the camera returns and talking works. No console errors. 192 tests.
Tradeoff: the page no longer works without the trained models, so a reviewer has to reproduce them (or read the screenshots and the video) before it does anything.
Also measured: the reply model repeated the person's sentence back word for word on 1 of 10 recognized dev lines (a garbled one). A repeat penalty of 1.2 or more stopped that and made the replies incoherent instead (it invented birthday presents and puppies), so it was not adopted.

## 2026-10-13: A better expression classifier for the live camera
Context: with the ViT's own FER2013 classifier on the live camera (the entry "A different face head for a live camera" above), a neutral face and a sad face both came out as joy at the webcam.
Checked first, on 259 dev clips with their MELD labels as a rough yardstick: the classifier says joy for 82 clips when 34 are joy, says joy on 27% of the true-neutral clips, and never says surprise (accuracy 0.286, weighted F1 0.272, macro F1 0.171). It was trained on small grey face crops, so we tried feeding it that: margins of 0.25, 0.10, 0.00 and -0.08 around the face box, each in colour and in grey. None helped (accuracy 0.22 to 0.29; the setting we had was the best of the eight). So the crop is not the problem, the model is.
Decision: `vision.expression_file`, the HSEmotion EfficientNet-B0 trained on AffectNet (in-the-wild colour photos, 8 classes; contempt is dropped and the other seven rescaled), run with ONNX Runtime on the CPU, with a margin of 0.10. On the same 259 clips: accuracy 0.266, weighted F1 0.275, macro F1 0.239, joy on true-neutral clips 14%, 42 clips called joy, and every label is used. It takes 11.6 ms a face on the CPU and no GPU memory. The ViT is loaded without its classifier again and only feeds our MELD head.
What this does and does not show: on MELD the new classifier is not more accurate overall, it is less biased towards joy. Whether it is right on a webcam is still not measured; there is no labelled webcam data here. In a headless browser with a dev clip as the camera it read faces as joy 0.52 and surprise 0.55 where the old one said joy 0.99.
Parity after the change: the live MELD-head path gives the cached top label on 120 of 120 dev clips, and `final_eval.py --split dev` prints the same seven rows.
Dependencies: `onnxruntime` 1.31.0, already installed as a dependency of faster-whisper and now listed. The model file (16 MB, Apache 2.0) is fetched by `scripts/fetch_models.py`. Parameters: 3,996,789 counted from the ONNX file; the total is 2,178,728,655, 36.3% of the cap.
Tradeoff: a third pretrained vision model in the system, used only for the camera's face view.
