import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset


class FakeOpt:
    pass


class TestBaseDataLoader:
    def test_init_stores_opt(self):
        opt = FakeOpt()
        loader = BaseDataLoader(opt)
        assert loader.opt is opt

    def test_load_data_returns_none(self):
        loader = BaseDataLoader(FakeOpt())
        assert loader.load_data() is None


class TestBaseDataset:
    def test_name(self):
        dataset = BaseDataset(FakeOpt())
        assert dataset.name() == 'BaseDataset'
