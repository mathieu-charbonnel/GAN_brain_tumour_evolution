import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.image_folder import is_image_file, make_dataset


class TestIsImageFile:
    def test_jpg_extension(self):
        assert is_image_file('image.jpg') is True

    def test_JPG_extension(self):
        assert is_image_file('image.JPG') is True

    def test_png_extension(self):
        assert is_image_file('image.png') is True

    def test_nii_extension(self):
        assert is_image_file('brain.nii') is True

    def test_nii_gz_extension(self):
        assert is_image_file('brain.nii.gz') is True

    def test_txt_extension(self):
        assert is_image_file('readme.txt') is False

    def test_py_extension(self):
        assert is_image_file('script.py') is False

    def test_bmp_extension(self):
        assert is_image_file('photo.bmp') is True

    def test_ppm_extension(self):
        assert is_image_file('photo.ppm') is True

    def test_jpeg_extension(self):
        assert is_image_file('photo.jpeg') is True


class TestMakeDataset:
    def test_finds_image_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['a.jpg', 'b.png', 'c.nii.gz', 'readme.txt']:
                open(os.path.join(tmpdir, name), 'w').close()
            result = make_dataset(tmpdir)
            assert len(result) == 3

    def test_returns_full_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.jpg')
            open(filepath, 'w').close()
            result = make_dataset(tmpdir)
            assert result[0] == filepath

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = make_dataset(tmpdir)
            assert len(result) == 0

    def test_walks_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, 'subdir')
            os.makedirs(subdir)
            open(os.path.join(subdir, 'nested.png'), 'w').close()
            result = make_dataset(tmpdir)
            assert len(result) == 1
