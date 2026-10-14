// The box on the right. In live mode it is the camera: the preview stays in the browser, small still
// images go to the local server for the face model, and what it found is drawn over the preview:
// a box around the face and the detector's five points (eyes, nose, mouth corners).
// In sample-clip mode it shows the video of the clip being played.
window.MELDRT_CAMERA = (function () {
  const body = document.getElementById("camera-body");
  const still = document.createElement("canvas");      // the picture that is sent, never shown
  const STILL_WIDTH = 480;
  let stream = null;
  let video = null;
  let overlay = null;
  let line = null;

  // one line of text: under the preview while the camera is on, otherwise instead of it
  function say(text) {
    if (stream) { line.textContent = text; return; }
    body.innerHTML = '<p class="dim"></p>';
    body.firstChild.textContent = text;
  }

  function newVideo() {
    const element = document.createElement("video");
    element.autoplay = true;
    element.muted = true;       // a clip's sound comes from the server, together with the robot's voice
    element.playsInline = true;
    return element;
  }

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    } catch (error) {
      stream = null;
      say("could not open the camera. check the browser permission. talking still works.");
      return false;
    }
    const frame = document.createElement("div");
    frame.className = "camera-frame";
    video = newVideo();
    video.srcObject = stream;
    overlay = document.createElement("canvas");
    frame.append(video, overlay);
    line = document.createElement("p");
    line.className = "dim small";
    body.replaceChildren(frame, line);
    return true;
  }

  function stop() {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video = null;
    say("camera off.");
  }

  // one still image from the camera as base64 JPEG, or null when the camera is off or not ready
  function grab() {
    if (!stream || !video || video.readyState < 2) return null;
    still.width = STILL_WIDTH;
    still.height = Math.round(STILL_WIDTH * video.videoHeight / video.videoWidth);
    still.getContext("2d").drawImage(video, 0, 0, still.width, still.height);
    return still.toDataURL("image/jpeg", 0.8).split(",")[1];
  }

  // draws what the face model found in the last image: positions come as shares of the picture
  function draw(face) {
    if (!stream) return;
    const width = overlay.width = video.clientWidth;
    const height = overlay.height = video.clientHeight;
    const pen = overlay.getContext("2d");
    pen.clearRect(0, 0, width, height);
    if (!face.found) { line.textContent = "no face in view."; return; }
    const [x, y, w, h] = face.box;
    pen.strokeStyle = pen.fillStyle = "#ffffff";
    pen.lineWidth = 1.5;
    pen.strokeRect(x * width, y * height, w * width, h * height);
    face.points.forEach(([px, py]) => { pen.beginPath(); pen.arc(px * width, py * height, 3.5, 0, 2 * Math.PI); pen.fill(); });
    line.textContent = "face: " + face.label + " " + face.confidence.toFixed(2);
  }

  // sample-clip mode: plays the video that belongs to the clip, if the server has one
  function showClip(name) {
    const clip = newVideo();
    clip.onerror = () => say("this sample has no video. the robot only hears it.");
    clip.src = "/clip-video/" + encodeURIComponent(name);
    body.replaceChildren(clip);
  }

  return { start, stop, grab, draw, say, showClip, isOn: () => stream !== null };
})();
