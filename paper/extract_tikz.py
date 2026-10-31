# -*- coding: utf-8 -*-
"""Extract the 3 inline TikZ figures from csar_net.tex into standalone .tex files."""
import io, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csar_net.tex')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tikz')
os.makedirs(OUT, exist_ok=True)

s = io.open(SRC, encoding='utf-8').read()

OPEN = '\\begin{tikzpicture}'
CLOSE = '\\end{tikzpicture}'

blocks = []
i = 0
while True:
    st = s.find(OPEN, i)
    if st == -1:
        break
    en = s.find(CLOSE, st)
    assert en != -1
    body = s[st:en + len(CLOSE)]
    body = body.rstrip()
    blocks.append(body)
    i = en + len(CLOSE)

assert len(blocks) == 3, 'expected 3 tikzpictures, got %d' % len(blocks)

PREAMBLE = (
    '\\documentclass[border=4pt]{standalone}\n'
    '\\usepackage{amsmath}\n'
    '\\usepackage{amssymb}\n'
    '\\usepackage{tikz}\n'
    '\\usetikzlibrary{positioning,arrows.meta,fit,calc,shadows}\n'
    '\\begin{document}\n'
)
END = '\n\\end{document}\n'

for name, body in zip(['fig_arch', 'fig_mechanism', 'fig_pipeline'], blocks):
    path = os.path.join(OUT, name + '.tex')
    io.open(path, 'w', encoding='utf-8').write(PREAMBLE + body + END)
    print('wrote', path, len(body), 'chars')
print('done')
