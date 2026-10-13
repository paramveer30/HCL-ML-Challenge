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

## 2026-10-12: Local scripts instead of a server
Context: the brief asks for an interaction-ready system that accepts input over time.
Decision: `live.py` (mic) and `replay.py` (MELD clip) stream 32 ms frames through the same pipeline. No web UI or network server.
Tradeoff: no remote client. A server would wrap the same pipeline without changing it.

## 2026-10-01: Class weights 1/sqrt(n) (pending approval)
Context: neutral is close to half of MELD; fear and disgust are a few percent each.
Decision: cross-entropy weights $w_c \propto 1/\sqrt{n_c}$.
Tradeoff: milder than full inverse frequency, so rare classes still get under-predicted somewhat, but the majority class isn't wrecked.
