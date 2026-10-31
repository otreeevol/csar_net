# -*- coding: utf-8 -*-
"""Build a pandoc-friendly copy of csar_net.tex:
   1. replace the CAS title/author block with plain, editable title/author/affiliation
   2. replace the 3 inline TikZ figures with the pre-rendered PNGs.
"""
import io, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'csar_net.tex')
DST = os.path.join(HERE, 'csar_net_pandoc.tex')

s = io.open(SRC, encoding='utf-8').read()

# ---- 1. strip the CAS title/author block ----
# Title, authors, affiliation and corresponding-author note are supplied to
# pandoc via --metadata so they land as a proper editable title block at the
# top of the .docx (a \begin{center} block in the body would otherwise be
# pushed after the abstract by pandoc's frontmatter handling).
OLD_TITLE_START = '%% Title'
OLD_TITLE_END = '\\cortext[cor1]{Corresponding author.}'
a = s.index(OLD_TITLE_START)
b = s.index(OLD_TITLE_END) + len(OLD_TITLE_END)
new_block = (
    '%% Title, authors and affiliation are injected via pandoc --metadata.\n'
)
s = s[:a] + new_block + s[b:]

# ---- 2. add an explicit References heading ----
# The CAS class auto-titles the bibliography, but pandoc's citeproc does not,
# so give it a plain \section*{References} heading before \bibliography.
s = s.replace(
    '\\bibliographystyle{model5-names}\n\t\\bibliography{csar_refs}',
    '\\section*{References}\n\\bibliography{csar_refs}',
)

# ---- 3. TikZ figures -> PNG ----
def replace_tikz(s, png, width='\\textwidth'):
    open_tag = '\\begin{tikzpicture}'
    close_tag = '\\end{tikzpicture}'
    wrap = '\\resizebox{\\textwidth}{!}{%'
    st = s.index(open_tag)
    # if a \resizebox wrapper opens immediately before the tikzpicture, back up to it
    head = s[max(0, st - 80):st]
    if wrap in head:
        st = st - len(head) + head.rindex(wrap)
    en = s.index(close_tag, st) + len(close_tag)
    # include a possible trailing '}' from a \resizebox wrapper
    if en < len(s) and s[en] == '}':
        en += 1
    repl = '\\includegraphics[width=%s]{%s}' % (width, png)
    return s[:st] + repl + s[en:]

# order matters: replace one at a time
s = replace_tikz(s, '_tikz/fig_arch.png', '\\textwidth')
s = replace_tikz(s, '_tikz/fig_mechanism.png', '\\textwidth')
s = replace_tikz(s, '_tikz/fig_pipeline.png', '0.72\\textwidth')

io.open(DST, 'w', encoding='utf-8').write(s)
print('wrote', DST)
print('remaining tikzpicture:', s.count('tikzpicture'))
