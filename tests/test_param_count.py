import struct

import pytest
import torch

from meldrt.eval.param_count import count_ct2, count_gguf, count_torch


def test_count_torch():
    assert count_torch(torch.nn.Linear(3, 2)) == 3 * 2 + 2


def ct2_string(text):
    data = text.encode() + b"\0"
    return struct.pack("<H", len(data)) + data


def ct2_variable(name, dims, n_bytes):
    return (ct2_string(name) + struct.pack("<B", len(dims)) + struct.pack(f"<{len(dims)}I", *dims)
            + struct.pack("<B", 0) + struct.pack("<I", n_bytes) + b"\0" * n_bytes)


def test_count_ct2_counts_shaped_variables_only(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(
        struct.pack("<I", 6) + ct2_string("WhisperSpec") + struct.pack("<I", 3) + struct.pack("<I", 3)
        + ct2_variable("encoder/weight", [4, 5], 80)
        + ct2_variable("encoder/bias", [5], 20)
        + ct2_variable("encoder/num_heads", [], 2)        # a setting, not a weight
    )
    assert count_ct2(path) == 4 * 5 + 5


def gguf_string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def gguf_tensor(name, dims):
    return (gguf_string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
            + struct.pack("<I", 0) + struct.pack("<Q", 0))


def test_count_gguf_skips_metadata_and_sums_tensor_shapes(tmp_path):
    metadata = (
        gguf_string("general.name") + struct.pack("<I", 8) + gguf_string("tiny")                 # a string
        + gguf_string("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 15)        # a uint32
        + gguf_string("tokenizer.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8)          # an array of strings
        + struct.pack("<Q", 2) + gguf_string("a") + gguf_string("bc")
        + gguf_string("tokenizer.scores") + struct.pack("<I", 9) + struct.pack("<I", 6)          # an array of floats
        + struct.pack("<Q", 3) + struct.pack("<3f", 0.1, 0.2, 0.3)
    )
    path = tmp_path / "tiny.gguf"
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 2) + struct.pack("<Q", 4) + metadata
        + gguf_tensor("token_embd.weight", [8, 100]) + gguf_tensor("output_norm.bias", [8])
    )
    assert count_gguf(path) == 8 * 100 + 8


def test_count_gguf_rejects_other_files(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"nope" + b"\0" * 32)
    with pytest.raises(ValueError):
        count_gguf(path)
