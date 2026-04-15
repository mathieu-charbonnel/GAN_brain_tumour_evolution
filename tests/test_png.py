import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util.png import encode


class TestPngEncode:
    def test_encode_returns_bytes(self):
        width, height = 2, 2
        buf = b'\xff\x00\x00' * (width * height)
        result = encode(buf, width, height)
        assert isinstance(result, bytes)

    def test_encode_starts_with_png_signature(self):
        width, height = 2, 2
        buf = b'\x00' * (width * height * 3)
        result = encode(buf, width, height)
        assert result[:8] == b'\x89PNG\r\n\x1a\n'

    def test_encode_assertion_on_wrong_size(self):
        try:
            encode(b'\x00' * 10, 2, 2)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    def test_encode_1x1(self):
        buf = b'\xff\x80\x40'
        result = encode(buf, 1, 1)
        assert result[:8] == b'\x89PNG\r\n\x1a\n'
        assert len(result) > 0
