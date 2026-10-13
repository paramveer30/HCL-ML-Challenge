"""The 7 MELD emotion labels and the label/index maps every model uses."""

LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}
