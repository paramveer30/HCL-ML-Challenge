// The whole page logic: one WebSocket in; the emotion line, the conversation and the numbers out.
const CONFIG = window.MELDRT_CONFIG;
const LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"];
const STATUS_TEXT = { idle: "ready", listening: "listening", hearing: "hearing you", thinking: "thinking", speaking: "speaking" };
const CAMERA = window.MELDRT_CAMERA;
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
let talking = false;
let held = false;           // is the talk button or the space bar down right now
let starting = false;       // true while the browser is setting up the microphone
let frameTimer = null;      // sends camera images while the talk button is held
let turns = new Map();      // turn_id -> { you, robot, timing }
let mic = null;             // set up on the first press of the talk button

// ---------- connection ----------
function connect() {
  socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + CONFIG.WS_PATH);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    send({ type: "mode", mock: $("mock").checked });
    // http://127.0.0.1:8000/?play=its_fine.wav replays that sample once the page is connected
    const sample = new URLSearchParams(location.search).get("play");
    if (sample && turns.size === 0) send({ type: "replay", clip: sample });
  };
  socket.onmessage = (message) => handle(JSON.parse(message.data));
  socket.onclose = () => { setStatus("idle"); setTimeout(connect, 1000); };
}
function send(control) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(control));
}

// ---------- events from the server ----------
function handle(event) {
  if (event.type === "status") setStatus(event.state);
  else if (event.type === "turn_start") turns.set(event.turn_id, { timing: {} });
  else if (event.type === "transcript") turnOf(event.turn_id).you = addLine("you", "you>", event.text);
  else if (event.type === "emotion") showEmotion(event);
  else if (event.type === "reply_delta") showReplyPiece(event);
  else if (event.type === "reply_done") { const robot = turnOf(event.turn_id).robot; if (robot) robot.classList.remove("streaming"); }
  else if (event.type === "timing") showTiming(event);
  else if (event.type === "notice") addLine("note", "#", event.text);
  // "speaking" events are not shown separately: the status line already says speaking
}

function setStatus(state) {
  $("status").dataset.state = state;
  $("status-text").textContent = STATUS_TEXT[state] || "ready";
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
function agreementText(event) {
  const against = [];
  if (event.disagree) against.push("tone");
  if (event.face_disagree) against.push("face");
  if (against.length === 0) return event.faces ? "words, tone and face agree" : "words and tone agree";
  if (against.length === 2) return "tone and face both point a different way from the words";
  return "words and " + against[0] + " point different ways";
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

async function talk(on) {
  held = on;
  if (on === talking || starting) return;
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
  // while the button is held, the camera (if it is on) sends one still image every CAMERA_FRAME_MS
  clearInterval(frameTimer);
  if (on) frameTimer = setInterval(() => { const jpeg = CAMERA.grab(); if (jpeg) send({ type: "frame", jpeg: jpeg }); }, CONFIG.CAMERA_FRAME_MS);
  $("talk").setAttribute("aria-pressed", String(on));
  $("talk").textContent = on ? "[ listening, let go when done ]" : "[ hold to talk ]";
  send({ type: "talk", on: on });
}

// ---------- controls ----------
const talkButton = $("talk");
talkButton.addEventListener("pointerdown", (event) => { event.preventDefault(); talk(true); });
window.addEventListener("pointerup", () => talk(false));
talkButton.addEventListener("pointerleave", () => talk(false));
window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat || ["SELECT", "INPUT", "SUMMARY"].includes(event.target.tagName)) return;
  event.preventDefault();
  talk(true);
});
window.addEventListener("keyup", (event) => { if (event.code === "Space") talk(false); });

$("play").addEventListener("click", () => {
  if (!$("clip").value) return;
  send({ type: "replay", clip: $("clip").value });
  // a real sample may come with its video: show it where the camera would be
  if (!$("mock").checked) CAMERA.showClip($("clip").value.replace(/\.wav$/, ".mp4"));
});
$("camera").addEventListener("change", async () => {
  if ($("camera").checked) $("camera").checked = await CAMERA.start();
  else CAMERA.stop();
});
$("mock").addEventListener("change", () => { send({ type: "mode", mock: $("mock").checked }); loadClips(); });
$("clear").addEventListener("click", () => {
  send({ type: "clear" });
  turns = new Map();
  log.replaceChildren();
  $("log-empty").hidden = false;
  $("emotion").textContent = "> waiting for a voice";
  $("emotion-note").textContent = "the emotion shows up here after a turn";
  $("faces-note").textContent = "tick camera to let the face model see you while you talk. images stay on this computer.";
  $("timing-rows").innerHTML = '<tr><td colspan="5" class="dim">n/a</td></tr>';
  drawMatrix(null);
});

async function loadClips() {
  const response = await fetch("/clips?mock=" + $("mock").checked);
  const names = (await response.json()).clips;
  $("clip").innerHTML = names.length
    ? names.map((name) => "<option>" + name + "</option>").join("")
    : '<option value="">no samples yet</option>';
  $("play").disabled = names.length === 0;
}

drawMatrix(null);
loadClips();
connect();
if (CONFIG.CAMERA_ON_AT_START) $("camera").click();
