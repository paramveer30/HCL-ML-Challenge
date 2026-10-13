// Runs on the audio thread. Converts the microphone signal to 16 kHz mono 16-bit PCM
// and posts it to the page in frames of exactly 512 samples (32 ms).
class MicFrames extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;   // input samples per output sample
    this.position = 0;                // where the next output sample falls in the input
    this.frame = new Int16Array(512);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    while (this.position < input.length) {
      // linear interpolation between the two nearest input samples
      const i = Math.floor(this.position);
      const next = Math.min(i + 1, input.length - 1);
      const sample = input[i] + (input[next] - input[i]) * (this.position - i);
      this.frame[this.filled++] = Math.max(-1, Math.min(1, sample)) * 32767;
      if (this.filled === 512) {
        this.port.postMessage(this.frame.buffer.slice(0));
        this.filled = 0;
      }
      this.position += this.step;
    }
    this.position -= input.length;
    return true;
  }
}

registerProcessor("mic-frames", MicFrames);
