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
| `state` | turn judged over | state ready | 600 | 576 | 713 |
| `first_token` | turn judged over | first reply token | 1,500 | 732 | 916 |
| `reply` | turn judged over | last reply token | 3,000 | 968 | 1,327 |
| `first_audio` | turn judged over | first spoken sound ready | 2,000 | 996 | 1,260 |

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
                                       |
                                 state (JSON)
                                       |
                                 reply prompt
                                       v
                           Qwen2.5-1.5B (Ollama, local)  --> streamed text
                                       |
                               Kokoro, per sentence  ------> spoken reply
```

> TODO (Param): 1 to 2 paragraphs walking one turn through the diagram. Mention: speech recognition and the audio model start together; the state keeps each model's own view and a conflict flag; the reply prompt checks in when the two disagree or confidence is low; the voice speaks sentence by sentence.

Parameters on the local inference path (`scripts/count_params.py`; counted from the loaded models and model files, not typed in):

| Component | Role | Parameters |
| --- | --- | ---: |
| Silero VAD | speech or not, per 32 ms frame | 462,594 |
| Whisper small.en (faster-whisper, int8) | speech to text | 241,761,792 |
| RoBERTa-base with emotion head | text emotion, fine-tuned by us | 124,651,015 |
| WavLM-base-plus | audio features, frozen | 94,381,936 |
| Audio head | audio emotion, trained by us | 198,676 |
| Qwen2.5-1.5B-Instruct (4-bit GGUF, Ollama) | reply text | 1,543,714,304 |
| Kokoro-82M | speech output | 81,810,022 |
| Kokoro voice embedding | speech output | 130,560 |
| spaCy `en_core_web_sm` | Kokoro's text front end | 1,570,514 |
| **Total** | | **2,088,681,413** |

The cap is 6,000,000,000, so the system uses 34.8% of it. Quantization changes how each number is stored, not how many there are, so the reply model counts at full size.

## 4. Data

MELD raw release: 9,989 / 1,109 / 2,610 utterances in train / dev / test, in 1,038 / 114 / 280 dialogues. About 11.9 hours of audio; the median clip is 2.5 s.

| | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| Utterances | 9,989 | 1,109 | 2,610 |
| Extracted WAVs | 9,988 | 1,112 | 2,615 |
| Clips under 0.3 s (no audio features) | 164 | 20 | 54 |
| Utterances with no audio features | 165 | 21 | 53 |

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

> TODO (Param): 2 paragraphs in your own words. (1) What the table shows: both models beat the baseline, text is the strong one, rare classes are weak. (2) Fusion: +0.020 weighted F1 on dev, -0.0015 on test. Why the dev gain did not carry over (alpha from a nearly flat curve on 1,109 rows; audio weaker on test than dev), that nothing was retuned after test, and what audio still contributes (the conflict flag).

## 6. Latency and resources

50 turns from dev clips, pushed through the live pipeline in 32 ms frames at real-time pace, with the voice preparing each sentence silently. Turn lengths ran from 0.5 to 7.4 s (median 1.7 s). Times are ms after the turn was judged over.

| Stage | p50 | p95 |
| --- | ---: | ---: |
| Speech recognition (Whisper) | 563 | 696 |
| Audio branch (WavLM + head), running at the same time | 455 | 521 |
| Text model (RoBERTa) | 15 | 20 |
| **State ready** | **576** | **713** |
| First reply token | 732 | 916 |
| Full reply | 968 | 1,327 |
| First spoken sound ready | 996 | 1,260 |

The state met its 600 ms target on 58% of turns; the other three clocks met theirs on every turn.

| Resource | Observed |
| --- | ---: |
| GPU memory before loading anything (other programs) | 1,858 MiB |
| GPU memory at peak, everything loaded | 5,365 MiB of 6,144 |
| GPU memory used by the system | 3,507 MiB |
| of which held by PyTorch (RoBERTa, WavLM, head, Kokoro) | 1,569 MiB |
| RAM at peak (our process plus Ollama) | 2,440 MiB |
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

Training: the text model took 503 s for 3 epochs; the audio head 25 s for 30 epochs; caching WavLM features for all 13,477 clips about 9 minutes; extracting audio about 10 minutes.

PyTorch models run in fp32: the GTX 1660 series has a known half-precision problem. Whisper runs as int8 inside CTranslate2 and the reply model as 4-bit GGUF inside Ollama.

## 8. Decisions and tradeoffs

> TODO (Param): pick the 5 or 6 decisions you most want to defend and give each 2 to 3 sentences: what you chose, what it cost. Candidates, all with measurements in `DECISIONS.md`: frozen WavLM with cached features; mean pooling over time; late fusion with one weight; class weights 1/sqrt(n); turn-based endpointing with 600 ms of silence; the 1.5B reply model; the voice on the GPU; no test access outside one script.

## 9. Limitations

> TODO (Param): write these in your own words. Facts to draw on, each measured or observed:
> - Fusion did not beat text alone on test.
> - Fear and disgust score around 0.2 F1.
> - One seed.
> - The accuracy of the live path is not measured: it uses Whisper's transcript, no dialogue context for a lone clip, and a generic speaker name. On 10 replayed dev clips the label matched 4 times.
> - MELD is acted TV audio with laugh tracks; several clips contain more or less speech than their transcript.
> - Pauses over 600 ms split one sentence into two turns; the voice detector can miss quiet speech at the start of a turn.
> - No barge-in: the microphone is ignored while the robot speaks.
> - The reply model is small and does not always follow the prompt; reply quality is not scored.
> - About 2% of utterances have no audio prediction (clips under 0.3 s).

## 10. Completed and intentionally left out

Completed: text emotion model with dialogue context; audio emotion model on frozen WavLM features; late fusion with an ablation on test; a streaming turn pipeline (voice detection, endpointing, speech recognition, classification, state, streamed reply, spoken reply); replay and live scripts; latency, memory, and parameter measurements.

Left out on purpose: the vision track and the three-modality extension; reinforcement learning; barge-in; a web UI or network server; a stacked fusion model; multiple seeds; an ablation of the dialogue context; a comparison of recognized against gold transcripts; rating of reply quality; an emotion-matched voice.

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
| PyTorch, Hugging Face transformers, scikit-learn, pandas, NumPy | training and evaluation | BSD / Apache 2.0 |

Versions are pinned in `requirements.txt`.

> TODO (Param): the brief asks to "identify important external or generated components and own their results". Decide what goes here about development tools, and write it yourself.

## 12. Time spent

> TODO (Param): the hours you actually spent, per block.
