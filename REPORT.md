# MELD-RT: report

Track: Text + Audio. Every number below came from a script in this repository, run on the machine in section 7. The full log of decisions and measurements, in order, is in `DECISIONS.md`.

## 1. Problem and framing

> TODO (Param): 1 short paragraph in your own words. What the robot should do, why tone matters ("great." said flatly vs happily), and why Text + Audio and not Text + Vision.

## 2. What "real-time" means here

> TODO (Param): 1 paragraph. The interaction is turn-based: a person speaks, pauses, and expects an answer about as fast as a person would give one. State the assumptions: one speaker, close microphone, English, a turn ends after 600 ms of silence, turns are capped at 15 s.

All clocks start when the turn is judged to be over, except `endpoint`, which is the silence waited before that.

Times in ms over 50 replayed turns (`scripts/bench_latency.py`).

| Clock | Starts | Ends | Target (p50) | p50 | p95 |
| --- | --- | --- | ---: | ---: | ---: |
| `endpoint` | last voiced frame | turn judged over | about 600 | 608 | 608 |
| `state` | turn judged over | state ready | 600 | 279 | 516 |
| `first_token` | turn judged over | first reply token | 1,500 | 352 | 635 |
| `reply` | turn judged over | last reply token | 3,000 | 497 | 834 |
| `first_audio` | turn judged over | first spoken sound ready | 2,000 | 582 | 844 |

## 3. System

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
                                 state (JSON)
                                       |
                                 reply prompt
                                       v
                           Qwen2.5-1.5B (Ollama, local)  --> streamed text
                                       |
                               Kokoro, per sentence  ------> spoken reply
```

> TODO (Param): 1 to 2 paragraphs walking one turn through the diagram. Mention: the audio model runs first, then speech recognition (side by side was slower on one GPU); the state keeps each model's own view and a conflict flag; the reply prompt checks in when the two disagree or confidence is low; the voice speaks sentence by sentence.

Parameters on the local inference path (`scripts/count_params.py`; counted from the loaded models and model files, not typed in):

| Component | Role | Parameters |
| --- | --- | ---: |
| Silero VAD | speech or not, per 32 ms frame | 462,594 |
| Whisper small.en (faster-whisper, 8-bit weights) | speech to text | 241,761,792 |
| RoBERTa-base with emotion head | text emotion, fine-tuned by us | 124,651,015 |
| WavLM-base-plus | audio features, frozen | 94,381,936 |
| Audio head | audio emotion, trained by us | 198,676 |
| Qwen2.5-1.5B-Instruct (4-bit GGUF, Ollama) | reply text | 1,543,714,304 |
| Kokoro-82M | speech output | 81,810,022 |
| Kokoro voice embedding | speech output | 130,560 |
| spaCy `en_core_web_sm` | Kokoro's text front end | 1,570,514 |
| YuNet face detector | finds faces in a picture | 53,121 |
| ViT-base face encoder | face features for our vision head, frozen | 85,798,656 |
| EfficientNet-B0 expression classifier (HSEmotion) | what a live camera face shows, frozen | 3,996,789 |
| Vision head | face emotion, trained by us | 198,676 |
| **Total** | | **2,178,728,655** |

The cap is 6,000,000,000, so the system uses 36.3% of it. Quantization changes how each number is stored, not how many there are, so the reply model counts at full size.

## 4. Data

MELD raw release: 9,989 / 1,109 / 2,610 utterances in train / dev / test, in 1,038 / 114 / 280 dialogues. About 11.9 hours of audio; the median clip is 2.5 s.

| | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| Utterances | 9,989 | 1,109 | 2,610 |
| Extracted WAVs | 9,988 | 1,112 | 2,615 |
| Clips under 0.3 s (no audio features) | 164 | 20 | 54 |
| Utterances with no audio features | 165 | 21 | 53 |
| Clips where no face was found in 8 frames | 61 | 9 | 18 |
| Utterances with no face features | 61 | 10 | 18 |

Neutral is 47.2% of train; fear and disgust are 268 and 271 utterances.

> TODO (Param): 1 paragraph on preprocessing in your own words: the text cleanup (apostrophes stored as control characters), the dialogue context (previous 2 utterances of the same dialogue, never the future), 16 kHz mono audio capped at 15 s, and why clips under 0.3 s get no audio features.

## 5. Results

All choices (best epoch, alpha) were made on dev. The test split was read once, by `scripts/final_eval.py`, after that script reproduced the saved dev numbers. One seed.

| Model | Dev weighted F1 | Dev macro F1 | Test weighted F1 | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority class (neutral) | 0.2523 | 0.0850 | 0.3127 | 0.0928 |
| Text only | 0.6150 | 0.4792 | 0.6384 | 0.4685 |
| Audio only | 0.5000 | 0.3582 | 0.4747 | 0.2927 |
| Fused (alpha 0.45) | 0.6346 | 0.4951 | 0.6369 | 0.4588 |

For the 53 test utterances with no audio features, "audio only" predicts neutral and "fused" uses text alone.

Per-class F1 on test:

| Emotion | n | Text only | Audio only | Fused |
| --- | ---: | ---: | ---: | ---: |
| neutral | 1256 | 0.793 | 0.664 | 0.793 |
| joy | 402 | 0.602 | 0.347 | 0.605 |
| surprise | 281 | 0.581 | 0.334 | 0.591 |
| anger | 345 | 0.441 | 0.328 | 0.443 |
| sadness | 208 | 0.415 | 0.243 | 0.395 |
| disgust | 68 | 0.237 | 0.057 | 0.182 |
| fear | 50 | 0.211 | 0.077 | 0.203 |

Fused confusion matrix on test (rows true, columns predicted):

```
           neutral  joy  surprise  anger  sadness  disgust  fear
neutral       1064   66        68     17       29        8     4
joy            100  240        40     14        6        2     0
surprise        36   24       203      9        5        1     3
anger           93   39        63    119        6       17     8
sadness         84   16        18     16       65        4     5
disgust         29    3         9     12        4       10     1
fear            23    3         5      5        6        0     8
```

The audio head's learned layer weights peak at WavLM layers 8 to 10 (0.150, 0.182, 0.121; equal weights would be 0.077).

### Vision extension

Added on Oct 13 after the core track was finished. For every clip, 8 evenly spaced frames; in each, the largest face found by a small detector (YuNet), cropped with a 25% margin; a frozen ViT trained for facial expressions turns the faces into features (the [CLS] token of all 13 layers, averaged over the faces); the same head design as audio is trained on them. A face was found in 99.4% / 99.2% / 99.3% of train / dev / test clips.

The test split was read a second time for these rows, with the vision head, the text + vision weight, and the three-model weights all fixed on dev first. The four rows of the first table came out identical in that run.

| Model | Dev weighted F1 | Dev macro F1 | Test weighted F1 | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Vision only | 0.3450 | 0.2073 | 0.3557 | 0.1853 |
| Text + vision (weight on text 0.45) | 0.6273 | 0.4795 | 0.6283 | 0.4496 |
| Text + audio + vision (weights 0.45 / 0.55 / 0.00) | 0.6346 | 0.4951 | 0.6369 | 0.4588 |

For utterances with no face, "vision only" predicts neutral, "text + vision" uses text alone, and the three-model row is the two-model fusion.

Per-class F1 on test:

| Emotion | n | Vision only | Text + vision | Text + audio + vision |
| --- | ---: | ---: | ---: | ---: |
| neutral | 1256 | 0.561 | 0.790 | 0.793 |
| joy | 402 | 0.294 | 0.611 | 0.605 |
| surprise | 281 | 0.066 | 0.576 | 0.591 |
| anger | 345 | 0.173 | 0.411 | 0.443 |
| sadness | 208 | 0.101 | 0.364 | 0.395 |
| disgust | 68 | 0.022 | 0.140 | 0.182 |
| fear | 50 | 0.078 | 0.255 | 0.203 |

What was measured on dev before choosing:
- The three-model weights were searched on a grid of 0.05 (231 triples). The best triple gives vision 0, which is the two-model fusion (0.6346). The best triple with any vision in it scores 0.6332 (0.50 / 0.20 / 0.30). The curve over the vision weight is flat between 0.629 and 0.635, so on MELD the face neither helps nor hurts once text and audio are there.
- Text + vision gains 0.012 over text alone on dev and loses 0.010 on test.
- The ViT's own expression classifier, with no MELD training, scores 0.257 / 0.171 on dev. The head trained on MELD scores 0.348 / 0.208 on the same rows, so the training is worth 0.09.
- Seven variants of the head (input normalization, learning rate, dropout, last layer only; two seeds each) all landed between 0.344 and 0.361, so the head is not what limits it.
- When the face model is at least 50% sure and disagrees with a text model that is also at least 50% sure (2.1% of dev rows), the words are right 48% of the time and the face 22%.
- When voice and face agree with each other against the words (150 dev rows), the words are right 47% of the time and the pair 31%; following the pair would lower dev weighted F1 from 0.635 to 0.601. So the fused label keeps following the dev-tuned weights, and those turns are flagged for the robot to check in instead (the fusion is right on only 55% of them).

### Confidence

The fused label is right 65.6% of the time on dev, but the fused probabilities said 0.51 on average: averaging a sharp text distribution with a flat audio one flattens the result. One temperature, fitted on dev by maximum likelihood, corrects the number without changing any label (`python -m meldrt.train.calibrate`). No F1 score depends on it.

| On dev | Accuracy | Mean confidence | Calibration error |
| --- | ---: | ---: | ---: |
| Fused, as averaged | 0.656 | 0.511 | 0.145 |
| Fused, temperature 0.70 | 0.656 | 0.634 | 0.038 |

After calibration, turns shown at 0.7 to 0.8 were right 76% of the time, at 0.8 to 0.9 82%, and above 0.9 91%. Fitted on the dialogues with an even number and checked on the odd ones (and the other way round) the temperature came out at 0.75 and 0.70, with mean confidence 0.61 and 0.64 against accuracy 0.68 and 0.64 on the held-out half. The temperature was not checked on test.

### Face view for a live camera

For pictures from a MELD clip the vision model is the head trained on MELD (the table above). A live camera is a different kind of picture: one person, facing it. There the face view comes from a facial expression classifier that was trained on in-the-wild colour photos (HSEmotion, EfficientNet-B0 on AffectNet, 4 million parameters, about 12 ms a face on the CPU).

We first used the classifier that ships with our ViT (trained on FER2013: small grey face crops). In front of a webcam it called a neutral face and a sad face joy. Measured on 259 dev clips with their MELD labels (a rough yardstick, since MELD labels the utterance and not the face, but the same one for both models):

| Expression classifier, no MELD training | Accuracy | Weighted F1 | Macro F1 | Says joy on true-neutral clips | Clips it calls joy (34 are) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT trained on FER2013 | 0.286 | 0.272 | 0.171 | 27% | 82 |
| EfficientNet-B0 trained on AffectNet (used) | 0.266 | 0.275 | 0.239 | 14% | 42 |

The second one is no more accurate overall on these clips, but it halves the joy mistakes and uses all seven labels (the first never said surprise). Tighter crops or grey input did not help the first model (eight variants, accuracy 0.22 to 0.29). Neither classifier beats our MELD-trained head on MELD (0.35 weighted F1), which is why clips keep that head. We have no labelled webcam data, so how good the live face view is on a real webcam is not measured. It cannot move any reported score: the vision weight in the fusion is 0.

> TODO (Param): 1 paragraph in your own words on why vision is weak on MELD (several faces, the largest is not always the speaker, the label is for the utterance and not the facial expression, TV lighting and cuts) and what it is still used for live (a third view in the state, the check-in when face and words disagree).

Checked afterwards on dev only (nothing was retrained for the results above): class weights $1/\sqrt{n_c}$ against no weights and full inverse frequency.

| Class weights | Text, dev weighted / macro F1 | Audio head, dev weighted / macro F1 (3 seeds) |
| --- | --- | --- |
| none | 0.608 / 0.456 | 0.474 to 0.479 / 0.303 to 0.317 |
| $1/\sqrt{n_c}$ (used) | 0.615 / 0.479 | 0.489 to 0.502 / 0.353 to 0.361 |
| $1/n_c$ | 0.590 / 0.474 | 0.419 to 0.435 / 0.314 to 0.339 |

Resampling the dev rows 2,000 times, the fusion gain over text (+0.020) has a 95% interval of +0.002 to +0.037.

> TODO (Param): 2 paragraphs in your own words. (1) What the table shows: both models beat the baseline, text is the strong one, rare classes are weak. (2) Fusion: +0.020 weighted F1 on dev, -0.0015 on test. Why the dev gain did not carry over (alpha from a nearly flat curve on 1,109 rows; audio weaker on test than dev), that nothing was retuned after test, and what audio still contributes (the conflict flag).

## 6. Latency and resources

50 turns from dev clips, pushed through the live pipeline in 32 ms frames at real-time pace, with the voice preparing each sentence silently. Turn lengths ran from 0.5 to 7.4 s (median 1.7 s). Times are ms after the turn was judged over.

| Stage | p50 | p95 |
| --- | ---: | ---: |
| Audio branch (WavLM + head) | 24 | 35 |
| Speech recognition (Whisper), after the audio branch | 212 | 436 |
| Text model (RoBERTa) | 13 | 20 |
| **State ready** | **253** | **489** |
| First reply token | 334 | 564 |
| Full reply | 450 | 745 |
| First spoken sound ready | 542 | 780 |

With the face model on and each clip's video fed in next to its audio (the configuration that ships, measured last, after the reply prompt and calibration changes), the same 50 clips gave: speech recognition 241 / 453, audio branch 28 / 37, vision head 1 / 1, **state 279 / 516**, first reply token 352 / 635, full reply 497 / 834, first sound 582 / 844. One video picture costs 67 / 94 ms (face detection on the CPU plus the ViT on the GPU), paid while the person is still speaking and not after the turn ends; 49 of the 50 turns had a face. State within 600 ms on 98% of turns, first sound within 1 s on 96%.

Without vision, the state met its 600 ms target on 98% of turns, the first reply token on 98%, the full reply on 100%, and the first sound on 98%. One turn of the 50 took about 2.6 s to reach the state.

The same benchmark before and after three changes made on Oct 13 (p50 / p95, same 50 clips, same prompt):

| Stage | Before | After |
| --- | ---: | ---: |
| Speech recognition | 579 / 919 | 212 / 436 |
| Audio branch | 469 / 550 | 24 / 35 |
| State ready | 596 / 934 | 253 / 489 |
| First reply token | 676 / 1,019 | 334 / 564 |
| Full reply | 808 / 1,178 | 450 / 745 |
| First spoken sound ready | 895 / 1,236 | 542 / 780 |

The three changes, each measured on its own first (`DECISIONS.md`): (1) Whisper with 8-bit weights and 32-bit arithmetic instead of 8-bit arithmetic, 325 to 204 ms alone on the GPU with the same word error rate; (2) the audio branch and speech recognition one after the other instead of side by side, 249 to 224 ms for the pair; (3) keeping the GPU busy during the 608 ms wait for the end of a turn. The GPU slows down while it sits idle during the speech: the same pair took 207 ms after 1 s of idle and 421 ms after 4 s. Running the audio model on half a second of silence once per frame during the wait brought it back to 228 ms.

| Resource | Observed |
| --- | ---: |
| GPU memory before loading anything (other programs) | 975 MiB |
| GPU memory at peak, everything loaded, vision included | 4,788 MiB of 6,144 |
| GPU memory used by the system | 3,813 MiB (3,371 without the face model) |
| of which held by PyTorch (RoBERTa, WavLM, ViT, heads, Kokoro) | 1,825 MiB |
| RAM at peak (our process plus Ollama) | 2,378 MiB |
| Model load time, already downloaded | about 4 s, plus the reply model |

A browser was open during the run. An earlier version of the benchmark read the memory counters twice a second from a second thread, which slowed the pipeline itself: the state p50 was 1,147 ms. Memory is now read between turns only.

> TODO (Param): 1 paragraph. Which targets were met, what dominates the delay, and that the 608 ms of endpointing silence comes before all of these.

## 7. Hardware

| | |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1660 Ti, 6 GB |
| CPU | AMD Ryzen 5 5500 (6 cores), 12 threads |
| RAM | 16 GB |
| OS | Windows 11 Pro |

Training: the text model took 503 s for 3 epochs; the audio head 25 s for 30 epochs; caching WavLM features for all 13,477 clips about 9 minutes; extracting audio about 10 minutes; caching face features for all 13,716 clips about 25 minutes (face detection in 6 CPU processes, the ViT on the GPU); the vision head 25 s.

PyTorch models run in fp32: the GTX 1660 series has a known half-precision problem. Whisper runs with 8-bit weights and 32-bit arithmetic (`int8_float32`) inside CTranslate2 and the reply model as 4-bit GGUF inside Ollama.

## 8. Decisions and tradeoffs

> TODO (Param): pick the 5 or 6 decisions you most want to defend and give each 2 to 3 sentences: what you chose, what it cost. Candidates, all with measurements in `DECISIONS.md`: frozen WavLM with cached features; mean pooling over time; late fusion with one weight; class weights 1/sqrt(n); turn-based endpointing with 600 ms of silence; the 1.5B reply model; the voice on the GPU; no test access outside one script.

## 9. Limitations

> TODO (Param): write these in your own words. Facts to draw on, each measured or observed:
> - Fusion did not beat text alone on test.
> - Fear and disgust score around 0.2 F1.
> - One seed.
> - The live path is less accurate than the offline scores. On the first 200 dev clips: offline fused 0.658 weighted / 0.525 macro F1; gold words with no dialogue context and the generic speaker name 0.615 / 0.535; the real live path (Whisper's words, no context) 0.523 / 0.379. So most of the loss comes from the recognized words (word error rate 29% on these TV clips), about 0.03 from the missing context, and almost none from the speaker name. The audio branch is identical live and offline (same top label on 248 of 248 clips).
> - MELD is acted TV audio with laugh tracks; several clips contain more or less speech than their transcript.
> - Pauses over 600 ms split one sentence into two turns: 11.5% of 400 dev clips came out as two or more turns, and 10.8% were not detected as a turn at all. A shorter wait would answer sooner but cut more: 14.2% at 500 ms, 18.0% at 400 ms. So the wait stayed at 600 ms.
> - In the terminal script the microphone is ignored while the robot speaks. In the web page, pressing talk again cuts the robot off.
> - The reply model is small (1.5B). With the rewritten prompt and stop phrases it picked up something the person said in 12 of 16 scripted turns (6 before) and used a stock assistant phrase in 1 (2 before, plus 2 where it read its instructions out). A 3.2B model answered more naturally but did not fit next to the other models on a 6 GB card: speech recognition time tripled. Reply quality on real conversations is not scored.
> - Vision: the face model is weak on MELD (0.356 weighted F1 on test, most predictions neutral, fear and disgust near zero) and gets no weight in the fused label. There is no active-speaker detection: the largest face is taken, and it is not always the speaker. The head was trained on TV frames, not on a webcam, and the live camera path was not scored. In the web page the camera sends pictures only while the talk button is held.
> - The reply model is small and does not always follow the prompt; reply quality is not scored.
> - About 2% of utterances have no audio prediction (clips under 0.3 s).

## 10. Completed and intentionally left out

Completed: text emotion model with dialogue context; audio emotion model on frozen WavLM features; late fusion with an ablation on test; a streaming turn pipeline (voice detection, endpointing, speech recognition, classification, state, streamed reply, spoken reply); replay and live scripts; a local web page that shows a turn as it happens; latency, memory, and parameter measurements; a check on dev of the weighting choices and of the accuracy lost in the live path; the three-modality extension (a vision emotion model, three-model fusion scored on test, and a live camera in the web page).

Left out on purpose: active-speaker detection for the vision branch; reinforcement learning; barge-in; a stacked fusion model; multiple seeds; an ablation of the dialogue context on the full dev set; rating of reply quality; an emotion-matched voice; a larger reply model.

## 11. External components

| Component | Used for | License |
| --- | --- | --- |
| MELD (declare-lab) | training and evaluation data | GPL-3.0 |
| RoBERTa-base | text model starting point | MIT |
| WavLM-base-plus (Microsoft) | audio features | TODO (Param): confirm on the model card |
| Silero VAD | voice activity detection | MIT |
| Whisper small.en through faster-whisper | speech recognition | MIT |
| Qwen2.5-1.5B-Instruct through Ollama | reply generation | Apache 2.0 |
| Kokoro-82M | speech output | Apache 2.0 |
| `trpakov/vit-face-expression` (ViT-base trained on FER2013) | face features | Apache 2.0 |
| HSEmotion `enet_b0_8_best_vgaf` (EfficientNet-B0 trained on AffectNet) through ONNX Runtime | facial expression for a live camera | Apache 2.0 |
| YuNet face detector (OpenCV Zoo) through OpenCV | finding faces | MIT (model), Apache 2.0 (OpenCV) |
| PyTorch, Hugging Face transformers, scikit-learn, pandas, NumPy | training and evaluation | BSD / Apache 2.0 |

Versions are pinned in `requirements.txt`.

> TODO (Param): the brief asks to "identify important external or generated components and own their results". Decide what goes here about development tools, and write it yourself.

## 12. Time spent

> TODO (Param): the hours you actually spent, per block.
