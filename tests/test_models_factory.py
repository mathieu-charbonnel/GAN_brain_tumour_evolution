import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.models import create_model


class FakeOpt:
    def __init__(self, model_name):
        self.model = model_name


class TestCreateModel:
    def test_invalid_model_raises(self):
        opt = FakeOpt('nonexistent_model')
        with pytest.raises(ValueError, match="not recognized"):
            create_model(opt)
