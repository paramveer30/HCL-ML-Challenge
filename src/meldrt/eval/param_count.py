"""Counts learned parameters from loaded models and from the model files other runtimes load."""
import struct


def count_torch(module):
    return sum(p.numel() for p in module.parameters())


def count_spacy(nlp):
    """Parameters of every component of a spaCy pipeline (Kokoro's text front end uses one)."""
    seen, total = set(), 0
    for _, component in nlp.pipeline:
        model = getattr(component, "model", None)
        if model is None:
            continue
        for node in model.walk():
            for name in node.param_names:
                if (node.id, name) not in seen:      # components can share a layer; count it once
                    seen.add((node.id, name))
                    total += node.get_param(name).size
    return total


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def count_ct2(path):
    """Tensor elements in a CTranslate2 model.bin, the file faster-whisper loads.

    Layout: version, spec name, revision, number of variables, then for each variable its
    name, shape, data type, and raw bytes. Variables with no shape are settings, not weights.
    """
    total = 0
    with open(path, "rb") as f:
        read(f, "I")                                  # binary version
        f.seek(read(f, "H"), 1)                       # spec name
        read(f, "I")                                  # revision
        for _ in range(read(f, "I")):
            f.seek(read(f, "H"), 1)                   # variable name
            dims = [read(f, "I") for _ in range(read(f, "B"))]
            read(f, "B")                              # data type
            f.seek(read(f, "I"), 1)                   # skip the raw bytes
            if dims:
                size = 1
                for d in dims:
                    size *= d
                total += size
    return total


# how many bytes each simple GGUF metadata value type takes
GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_STRING, GGUF_ARRAY = 8, 9


def skip_gguf_value(f, value_type):
    if value_type == GGUF_STRING:
        f.seek(read(f, "Q"), 1)
    elif value_type == GGUF_ARRAY:
        item_type, count = read(f, "I"), read(f, "Q")
        if item_type in GGUF_SIZES:
            f.seek(GGUF_SIZES[item_type] * count, 1)
        else:
            for _ in range(count):
                skip_gguf_value(f, item_type)
    else:
        f.seek(GGUF_SIZES[value_type], 1)


def count_gguf(path):
    """Tensor elements in a GGUF file, the format Ollama stores the reply model in.

    The header lists every tensor with its shape, so nothing has to be loaded. Quantizing to
    4 bits changes how each number is stored, not how many numbers there are.
    """
    total = 0
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        read(f, "I")                                  # format version
        n_tensors, n_metadata = read(f, "Q"), read(f, "Q")
        for _ in range(n_metadata):
            f.seek(read(f, "Q"), 1)                   # key
            skip_gguf_value(f, read(f, "I"))
        for _ in range(n_tensors):
            f.seek(read(f, "Q"), 1)                   # tensor name
            size = 1
            for _ in range(read(f, "I")):
                size *= read(f, "Q")
            total += size
            read(f, "I")                              # data type
            read(f, "Q")                              # offset of the data
    return total
