// The page logic: one WebSocket to the local server; the conversation, the numbers and the face tracker out.
// Two modes. Live: your microphone and camera. Sample clips: a recorded clip, its sound and its video.
const CONFIG = window.MELDRT_CONFIG;
const CAMERA = window.MELDRT_CAMERA;
const LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"];
const STATUS_TEXT = {
  idle: "ready", loading: "loading the models (about 30 s the first time)",
  unavailable: "the models could not be loaded: see the conversation box",
  listening: "listening", hearing: "hearing", thinking: "thinking", speaking: "speaking",
};
const MODE_NOTE = {
  live: "live: you talk to the robot. it hears your microphone and sees your camera.",
  samples: "sample clips: the robot watches a recorded clip instead of you. same models, nothing made up.",
};
const FACES_NOTE = {
  live: "the face model sees you while the camera is on. images stay on this computer.",
  samples: "pick a clip and press play: its video shows here.",
};
const TIMING_STAGES = ["transcript", "emotion", "first_word", "first_audio"];
const BAR_CELLS = 5;

const $ = (id) => document.getElementById(id);
const log = $("log");
const logBody = $("log-body");     // the part of the conversation box that scrolls
$("author").textContent = CONFIG.AUTHOR;
$("email").textContent = CONFIG.EMAIL;
$("title").textContent = CONFIG.TITLE;
$("org").textContent = CONFIG.ORG;

let socket = null;
let mode = "live";          // "live" or "samples"
let ready = false;          // false while the models load, and if they could not be loaded
let talking = false;
let held = false;           // is the talk button or the space bar down right now
let starting = false;       // true while the browser is setting up the microphone
let faceAskedAt = 0;        // when the last camera image was sent; 0 once its answer is back
let turns = new Map();      // turn_id -> { you, robot, timing }
let mic = null;             // set up on the first press of the talk button

// ---------- connection ----------
function connect() {
  socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + CONFIG.WS_PATH);
  socket.binaryType = "arraybuffer";
  socket.onmessage = (message) => handle(JSON.parse(message.data));
  socket.onclose = () => { setStatus("loading"); setTimeout(connect, 1000); };
}
function send(control) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(control));
}

// ---------- events from the server ----------
function handle(event) {
  if (event.type === "status") setStatus(event.state);
  else if (event.type === "turn_start") turns.set(event.turn_id, { timing: {} });
  else if (event.type === "transcript") turnOf(event.turn_id).you = addLine("you", mode === "live" ? "you>" : "clip>", event.text);
  else if (event.type === "emotion") showEmotion(event);
  else if (event.type === "reply_delta") showReplyPiece(event);
  else if (event.type === "reply_done") endReply(turnOf(event.turn_id).robot);
  else if (event.type === "timing") showTiming(event);
  else if (event.type === "face") { faceAskedAt = 0; CAMERA.draw(event); }
  else if (event.type === "notice") addLine("note", "#", event.text);
}

function setStatus(state) {
  $("status").dataset.state = state;
  $("status-text").textContent = STATUS_TEXT[state] || "ready";
  ready = state !== "loading" && state !== "unavailable";
  showControls();
  // a new turn began: whatever the robot was still saying has been cut off
  if (state === "listening") document.querySelectorAll(".line.robot.streaming").forEach(endReply);
}

function turnOf(id) {
  if (!turns.has(id)) turns.set(id, { timing: {} });
  return turns.get(id);
}

function addLine(kind, who, text) {
  $("log-empty").hidden = true;
  const item = document.createElement("li");
  item.className = "line " + kind;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  const words = document.createElement("span");
  words.className = "words";
  words.textContent = text;
  item.append(label, words);
  log.appendChild(item);
  logBody.scrollTop = logBody.scrollHeight;
  return item;
}

// what the three models say about each other, in a few words
function topOf(probs) {
  return LABELS.reduce((best, label) => (probs[label] > probs[best] ? label : best), LABELS[0]);
}

function agreementText(event) {
  const words = topOf(event.text), voice = topOf(event.audio), face = topOf(event.vision);
  if (event.disagree && event.face_disagree) {
    return voice === face
      ? "words say " + words + ", but voice and face both say " + voice
      : "words say " + words + ", but voice says " + voice + " and face says " + face;
  }
  if (event.disagree) return "words say " + words + ", but voice says " + voice;
  if (event.face_disagree) return "words say " + words + ", but face says " + face;
  return event.faces ? "words, voice and face do not disagree" : "words and voice do not disagree";
}

function showEmotion(event) {
  $("emotion").textContent = "> " + event.top + " " + event.confidence.toFixed(2);
  $("emotion-note").textContent = agreementText(event);
  $("faces-note").textContent = event.faces
    ? "the face model saw a face in " + event.faces + " image" + (event.faces === 1 ? "" : "s") + " during this turn."
    : "no face seen during this turn: the result is from words and voice.";
  drawMatrix(event);

  const you = turnOf(event.turn_id).you;
  if (!you) return;
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = event.top + " " + event.confidence.toFixed(2);
  if (event.disagree || event.face_disagree) {
    const alert = document.createElement("span");
    alert.className = "alert";
    alert.textContent = "! " + agreementText(event);
    meta.appendChild(alert);
  }
  you.appendChild(meta);
}

function showReplyPiece(event) {
  const turn = turnOf(event.turn_id);
  if (!turn.robot) { turn.robot = addLine("robot", "robot>", ""); turn.robot.classList.add("streaming"); }
  turn.robot.querySelector(".words").textContent += event.text;
  logBody.scrollTop = logBody.scrollHeight;
}

function endReply(line) {
  if (line) line.classList.remove("streaming");
}

function showTiming(event) {
  turnOf(event.turn_id).timing[event.stage] = event.ms;
  const recent = [...turns.entries()].filter(([, turn]) => Object.keys(turn.timing).length).slice(-3).reverse();
  $("timing-rows").innerHTML = recent.map(([id, turn]) =>
    "<tr><td>" + id + "</td>" + TIMING_STAGES.map((stage) =>
      "<td>" + (turn.timing[stage] === undefined ? "n/a" : turn.timing[stage]) + "</td>").join("") + "</tr>").join("");
}

// ---------- the numbers: one row per emotion, one column per model ----------
function drawMatrix(event) {
  const columns = ["text", "audio", "vision", "fused"];
  // the winning emotion in each column, or none before the first turn
  const tops = columns.map((column) => event
    ? LABELS.reduce((best, label) => (event[column][label] > event[column][best] ? label : best), LABELS[0]) : null);
  $("matrix-rows").innerHTML = LABELS.map((label) => "<tr><th>" + label + "</th>" + columns.map((column, i) => {
    const p = event ? event[column][label] : 0;
    const on = Math.round(p * BAR_CELLS);
    return '<td class="' + (tops[i] === label ? "is-top" : "") + '"><span class="on">' + "█".repeat(on) + "</span>" +
      '<span class="off">' + "░".repeat(BAR_CELLS - on) + "</span> " + p.toFixed(2) + "</td>";
  }).join("") + "</tr>").join("");
}

// ---------- microphone: 16 kHz mono 16-bit PCM, 512 samples per frame ----------
async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const sendFrame = (buffer) => { if (talking && socket.readyState === WebSocket.OPEN) socket.send(buffer); };
  try {
    await context.audioWorklet.addModule("mic-worklet.js");
    const node = new AudioWorkletNode(context, "mic-frames");
    node.port.onmessage = (message) => sendFrame(message.data);
    source.connect(node);
  } catch (error) {
    // older browsers: the same conversion on the main thread
    const node = context.createScriptProcessor(4096, 1, 1);
    const step = context.sampleRate / 16000;
    let position = 0, frame = new Int16Array(512), filled = 0;
    node.onaudioprocess = (audio) => {
      const input = audio.inputBuffer.getChannelData(0);
      while (position < input.length) {
        const i = Math.floor(position), next = Math.min(i + 1, input.length - 1);
        const sample = input[i] + (input[next] - input[i]) * (position - i);
        frame[filled++] = Math.max(-1, Math.min(1, sample)) * 32767;
        if (filled === 512) { sendFrame(frame.buffer.slice(0)); filled = 0; }
        position += step;
      }
      position -= input.length;
    };
    source.connect(node);
    node.connect(context.destination);
  }
  return context;
}

// hold to talk: live mode only
async function talk(on) {
  held = on;
  if (on === talking || starting || (on && (mode !== "live" || !ready))) return;
  if (on && !mic) {
    starting = true;
    try { mic = await startMic(); }
    catch (error) { mic = null; addLine("note", "#", "could not reach the microphone. check the browser permission and try again."); }
    starting = false;
    // the button was let go while the browser was still asking for the microphone
    if (!held) return;
  }
  if (on && mic && mic.state === "suspended") await mic.resume();
  talking = on;
  $("talk").setAttribute("aria-pressed", String(on));
  $("talk").textContent = on ? "[ listening, let go when done ]" : "[ hold to talk ]";
  send({ type: "talk", on: on });
}

// ---------- camera: one still image at a time to the face model, in live mode while the camera is on ----------
setInterval(() => {
  // the next image only goes out once the answer to the last one is back (or a second has passed)
  if (mode !== "live" || !ready || !CAMERA.isOn() || (faceAskedAt && Date.now() - faceAskedAt < 1000)) return;
  const jpeg = CAMERA.grab();
  if (!jpeg) return;
  faceAskedAt = Date.now();
  send({ type: "frame", jpeg: jpeg });
}, CONFIG.CAMERA_FRAME_MS);

async function setCamera(on) {
  if (on) on = await CAMERA.start();
  else CAMERA.stop();
  choose("camera-on", "camera-off", on);
}

// ---------- controls ----------
// two buttons that work as one switch: the pressed one is the current choice
function choose(yesId, noId, yes) {
  $(yesId).setAttribute("aria-pressed", String(yes));
  $(noId).setAttribute("aria-pressed", String(!yes));
}

// which buttons can be used right now
function showControls() {
  $("talk").disabled = !ready;
  $("play").disabled = !ready || !$("clip").value;
}

// empties the page and makes the robot forget what was said
function clearConversation() {
  send({ type: "clear" });
  turns = new Map();
  log.replaceChildren();
  $("log-empty").hidden = false;
  $("emotion").textContent = "> waiting for a turn";
  $("emotion-note").textContent = "the emotion shows up here after a turn";
  $("faces-note").textContent = FACES_NOTE[mode];
  $("timing-rows").innerHTML = '<tr><td colspan="5" class="dim">n/a</td></tr>';
  drawMatrix(null);
}

async function setMode(next) {
  if (talking) await talk(false);
  // each mode is its own conversation: what a clip said is not context for what you say, and the other way round
  if (next !== mode) clearConversation();
  mode = next;
  const live = mode === "live";
  choose("mode-live", "mode-samples", live);
  $("mode-note").textContent = MODE_NOTE[mode];
  $("live-block").hidden = !live;
  $("samples-block").hidden = live;
  $("camera-switch").hidden = !live;
  $("camera-title").textContent = live ? "camera" : "sample clip";
  $("faces-note").textContent = FACES_NOTE[mode];
  // live: your camera. sample clips: the box belongs to the clip's video
  await setCamera(live && CONFIG.CAMERA_ON_IN_LIVE);
  if (!live) CAMERA.say("no clip playing.");
}

const talkButton = $("talk");
talkButton.addEventListener("pointerdown", (event) => { event.preventDefault(); talk(true); });
window.addEventListener("pointerup", () => talk(false));
talkButton.addEventListener("pointerleave", () => talk(false));
window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat || ["SELECT", "INPUT"].includes(event.target.tagName)) return;
  event.preventDefault();
  talk(true);
});
window.addEventListener("keyup", (event) => { if (event.code === "Space") talk(false); });

$("play").addEventListener("click", () => {
  if (mode !== "samples" || !ready || !$("clip").value) return;
  send({ type: "replay", clip: $("clip").value });
  CAMERA.showClip($("clip").value.replace(/\.wav$/, ".mp4"));
});
$("mode-live").addEventListener("click", () => setMode("live"));
$("mode-samples").addEventListener("click", () => setMode("samples"));
$("camera-on").addEventListener("click", () => setCamera(true));
$("camera-off").addEventListener("click", () => setCamera(false));
$("clear").addEventListener("click", clearConversation);

async function loadClips() {
  const response = await fetch("/clips");
  const names = (await response.json()).clips;
  $("clip").innerHTML = names.length
    ? names.map((name) => "<option>" + name + "</option>").join("")
    : '<option value="">no samples yet: see ui/README.md</option>';
  showControls();
}

drawMatrix(null);
loadClips();
connect();
setMode("live");
