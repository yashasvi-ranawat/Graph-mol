import sys

sys.path.append("../app/")

import numpy as np
import predict
from _pytest.monkeypatch import MonkeyPatch


def test_predict():
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(predict, "MODEL_PATH", "../app/model.ckpt")
    with open("dsgdb9nsd_065161.xyz") as fio:
        file_ = fio.readlines()

    dict_ = predict.predict(file_)

    print(dict_)

    assert isinstance(dict_["G"], float)
    assert isinstance(dict_["gap"], float)
    assert len(dict_["charge"].shape) == 1
    assert isinstance(dict_["G_loss"], float)
    assert isinstance(dict_["gap_loss"], float)
    assert isinstance(dict_["charge_loss"], float)
