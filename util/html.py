import os
from typing import List

import dominate
from dominate.tags import a, br, h3, img, meta, p, table, td, tr


class HTML:
    def __init__(self, web_dir: str, title: str, reflesh: int = 0) -> None:
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self) -> str:
        return self.img_dir

    def add_header(self, text: str) -> None:
        with self.doc:
            h3(text)

    def add_table(self, border: int = 1) -> None:
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims: List[str], txts: List[str], links: List[str], width: int = 400) -> None:
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self) -> None:
        html_file = '%s/index.html' % self.web_dir
        with open(html_file, 'wt') as f:
            f.write(self.doc.render())


if __name__ == '__main__':
    html_page = HTML('web/', 'test_html')
    html_page.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html_page.add_images(ims, txts, links)
    html_page.save()
