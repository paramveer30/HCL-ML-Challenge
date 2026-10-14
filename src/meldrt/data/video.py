"""The video side of a clip: a few evenly spaced frames, and the largest face in each, cropped square."""
import cv2
import numpy as np


def sample_indexes(n_frames, k):
    """k frame numbers spread evenly over n_frames: the middle of each of k equal slices. All of them if n_frames <= k."""
    if n_frames <= k:
        return list(range(n_frames))
    return [int((i + 0.5) * n_frames / k) for i in range(k)]


def read_timed_frames(path, k, max_seconds):
    """Up to k (seconds into the clip, BGR uint8 frame) pairs spread evenly over the first max_seconds of a video file."""
    video = cv2.VideoCapture(str(path))
    fps = video.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = min(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), int(max_seconds * fps))
    wanted = set(sample_indexes(n_frames, k))
    frames = []
    for i in range(n_frames):
        # grab() moves to the next frame; only the frames we keep are fully decoded
        if not video.grab():
            break
        if i in wanted:
            ok, frame = video.retrieve()
            if ok:
                frames.append((i / fps, frame))
    video.release()
    return frames


def read_frames(path, k, max_seconds):
    """Up to k frames (BGR, uint8) spread evenly over the first max_seconds of a video file."""
    return [frame for _, frame in read_timed_frames(path, k, max_seconds)]


def square_box(box, margin, frame_width, frame_height):
    """box: (x, y, w, h) of a face. Returns (left, top, right, bottom) of a square around its center,
    wider than the face by margin on every side, and cut off at the edges of the frame."""
    x, y, w, h = box
    half = max(w, h) * (1 + 2 * margin) / 2
    center_x, center_y = x + w / 2, y + h / 2
    left, top = max(0, int(center_x - half)), max(0, int(center_y - half))
    right, bottom = min(frame_width, int(center_x + half)), min(frame_height, int(center_y + half))
    return left, top, right, bottom


def crop_face(frame, box, margin, size):
    """Cuts the face out of a BGR frame and returns it as a size x size RGB image, uint8."""
    left, top, right, bottom = square_box(box, margin, frame.shape[1], frame.shape[0])
    face = cv2.resize(frame[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


class FaceFinder:
    """YuNet face detector. Picks the largest face: in a TV shot that is usually the person speaking."""

    def __init__(self, detector_file, min_score, min_face_px, detect_width):
        self.detector = cv2.FaceDetectorYN.create(str(detector_file), "", (320, 320), min_score)
        self.min_face_px = min_face_px
        self.detect_width = detect_width

    def largest_face(self, frame):
        """Returns (box, points) for the largest face in a BGR frame, or None if there is no face big enough.

        box: (x, y, w, h) in pixels. points: the detector's 5 landmarks as (x, y) pairs:
        the two eyes, the tip of the nose, and the two corners of the mouth.
        """
        # faces are looked for on a smaller copy of a wide frame (much faster on 1280 x 720);
        # everything is scaled back, so the crop still comes from the full-size frame
        scale = min(1.0, self.detect_width / frame.shape[1])
        small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return None
        # one row per face: x, y, w, h, then 5 landmarks as x, y pairs, then the score
        face = [float(value) / scale for value in max(faces, key=lambda row: row[2] * row[3])[:14]]
        box = tuple(face[:4])
        if min(box[2], box[3]) < self.min_face_px:
            return None
        return box, [(face[i], face[i + 1]) for i in range(4, 14, 2)]


def face_crops(frames, finder, margin, size):
    """One face crop per frame that has a face. Returns uint8 (n, size, size, 3), where n can be 0."""
    crops = []
    for frame in frames:
        found = finder.largest_face(frame)
        if found is not None:
            crops.append(crop_face(frame, found[0], margin, size))
    return np.stack(crops) if crops else np.zeros((0, size, size, 3), dtype=np.uint8)
