import numpy as np
import pandas as pd

from meldrt.data.features import load_features


def test_load_features_marks_rows_without_a_file(tmp_path):
    cfg = {"data": {"feature_root": str(tmp_path)}}
    (tmp_path / "dev").mkdir()
    np.save(tmp_path / "dev" / "dia0_utt0.npy", np.full((13, 768), 1, dtype=np.float16))
    np.save(tmp_path / "dev" / "dia0_utt2.npy", np.full((13, 768), 3, dtype=np.float16))
    df = pd.DataFrame({"wav_path": ["x/dev/dia0_utt0.wav", "x/dev/dia0_utt1.wav", "x/dev/dia0_utt2.wav"]})

    feats, has_audio = load_features(df, "dev", cfg)

    assert has_audio.tolist() == [True, False, True]
    assert feats.shape == (2, 13, 768)
    assert feats.dtype == np.float32
    assert feats[0, 0, 0] == 1 and feats[1, 0, 0] == 3   # row order is kept
