import numpy as np

from meldrt.data.video import crop_face, face_crops, sample_indexes, square_box


def test_sample_indexes_takes_the_middle_of_equal_slices():
    assert sample_indexes(80, 8) == [5, 15, 25, 35, 45, 55, 65, 75]
    assert sample_indexes(24, 3) == [4, 12, 20]


def test_a_short_clip_gives_every_frame():
    assert sample_indexes(5, 8) == [0, 1, 2, 3, 4]
    assert sample_indexes(0, 8) == []


def test_square_box_adds_the_margin_on_every_side():
    # a 100 x 100 face at (200, 100) with a 25% margin becomes 150 x 150 around the same center
    assert square_box((200, 100, 100, 100), 0.25, 1280, 720) == (175, 75, 325, 225)


def test_square_box_uses_the_longer_side_and_stops_at_the_frame_edge():
    assert square_box((0, 0, 40, 80), 0.0, 1280, 720) == (0, 0, 60, 80)
    assert square_box((1200, 660, 80, 60), 0.5, 1280, 720) == (1160, 610, 1280, 720)


def test_crop_face_returns_a_square_rgb_image():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 0] = 255                       # all blue in BGR
    crop = crop_face(frame, (600, 300, 100, 120), 0.25, 224)
    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8
    assert crop[0, 0].tolist() == [0, 0, 255]  # blue is last in RGB


class FakeFinder:
    """Finds a face in every second frame."""

    def __init__(self):
        self.calls = 0

    def largest_face(self, frame):
        self.calls += 1
        return ((100, 100, 80, 80), []) if self.calls % 2 else None


def test_face_crops_keeps_only_frames_with_a_face():
    frames = [np.zeros((360, 640, 3), dtype=np.uint8)] * 5
    assert face_crops(frames, FakeFinder(), 0.25, 224).shape == (3, 224, 224, 3)


def test_no_faces_gives_an_empty_array_of_the_right_shape():
    class Blind:
        def largest_face(self, frame):
            return None

    crops = face_crops([np.zeros((360, 640, 3), dtype=np.uint8)], Blind(), 0.25, 224)
    assert crops.shape == (0, 224, 224, 3) and crops.dtype == np.uint8
