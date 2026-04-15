import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset


class TestBaseDataLoader:
    def test_init(self):
        loader = BaseDataLoader()
        assert loader is not None

    def test_load_data_returns_none(self):
        loader = BaseDataLoader()
        assert loader.load_data() is None

    def test_initialize_stores_opt(self):
        loader = BaseDataLoader()

        class FakeOpt:
            pass

        opt = FakeOpt()
        loader.initialize(opt)
        assert loader.opt is opt


class TestBaseDataset:
    def test_name(self):
        dataset = BaseDataset()
        assert dataset.name() == 'BaseDataset'

    def test_initialize_no_error(self):
        dataset = BaseDataset()

        class FakeOpt:
            pass

        dataset.initialize(FakeOpt())
