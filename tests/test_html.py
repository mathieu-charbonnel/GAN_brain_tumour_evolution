import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from util.html import HTML


class TestHTML:
    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            web_dir = os.path.join(tmpdir, 'web')
            html_page = HTML(web_dir, 'Test Page')
            assert os.path.isdir(web_dir)
            assert os.path.isdir(os.path.join(web_dir, 'images'))

    def test_get_image_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            web_dir = os.path.join(tmpdir, 'web')
            html_page = HTML(web_dir, 'Test Page')
            assert html_page.get_image_dir() == os.path.join(web_dir, 'images')

    def test_save_creates_index_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            web_dir = os.path.join(tmpdir, 'web')
            html_page = HTML(web_dir, 'Test Page')
            html_page.add_header('Test Header')
            html_page.save()
            index_path = os.path.join(web_dir, 'index.html')
            assert os.path.isfile(index_path)

    def test_save_contains_title(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            web_dir = os.path.join(tmpdir, 'web')
            html_page = HTML(web_dir, 'My Title')
            html_page.save()
            with open(os.path.join(web_dir, 'index.html'), 'r') as f:
                content = f.read()
            assert 'My Title' in content

    def test_add_header_in_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            web_dir = os.path.join(tmpdir, 'web')
            html_page = HTML(web_dir, 'Test')
            html_page.add_header('Section Header')
            html_page.save()
            with open(os.path.join(web_dir, 'index.html'), 'r') as f:
                content = f.read()
            assert 'Section Header' in content
