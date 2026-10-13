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

