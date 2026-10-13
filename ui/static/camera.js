// The camera box. Off until the camera switch is ticked. The preview stays in the browser;
// while the talk button is held, small still images go to the local server for the face model.
// When a sample with a video file is replayed, the box shows that video instead.
window.MELDRT_CAMERA = (function () {
  const body = document.getElementById("camera-body");
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 240;
  let stream = null;
  let video = null;

  function say(text) {
    body.innerHTML = '<p class="dim"></p>';
    body.firstChild.textContent = text;
  }

  function newVideo() {
    video = document.createElement("video");
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    body.replaceChildren(video);
    return video;
  }

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      newVideo().srcObject = stream;
      return true;
    } catch (error) {
      say("could not open the camera. check the browser permission.");
      return false;
    }
  }

  function stop() {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video = null;
    say("off");
  }

  // one still image from the camera as base64 JPEG, or null when the camera is off or not ready
  function grab() {
    if (!stream || !video || video.readyState < 2) return null;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
  }

  // plays the video that belongs to a replayed sample, if the server has one; the camera comes back after it
  function showClip(name) {
    if (stream) return;
    const clip = newVideo();
    clip.onerror = () => say("off");
    clip.onended = () => say("off");
    clip.src = "/clip-video/" + encodeURIComponent(name);
  }

  say("off");
  return { start, stop, grab, showClip, isOn: () => stream !== null };
})();
