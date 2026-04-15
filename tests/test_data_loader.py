import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.base_data_loader import BaseDataLoader


class FakeOpt:
    pass


class TestBaseDataLoaderInit:
    def test_load_data_default_none(self):
        loader = BaseDataLoader(FakeOpt())
        assert loader.load_data() is None

    def test_init_sets_opt(self):
        opt = FakeOpt()
        loader = BaseDataLoader(opt)
        assert hasattr(loader, 'opt')
        assert loader.opt is opt
