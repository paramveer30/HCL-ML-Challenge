from meldrt.labels import ID2LABEL, LABEL2ID, LABELS


def test_seven_labels_in_spec_order():
    assert LABELS == ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]


def test_maps_round_trip():
    for i, name in enumerate(LABELS):
        assert LABEL2ID[name] == i
        assert ID2LABEL[i] == name
    assert len(LABEL2ID) == len(ID2LABEL) == 7
