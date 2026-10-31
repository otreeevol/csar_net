# -*- coding: utf-8 -*-
"""Build a pandoc-friendly copy of csar_net.tex:
   1. replace the CAS title/author block with plain, editable title/author/affiliation
   2. replace the 3 inline TikZ figures with the pre-rendered PNGs.
"""
import io, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'csar_net.tex')
DST = os.path.join(HERE, 'csar_net_pandoc.tex')

s = io.open(SRC, encoding='utf-8').read()

# ---- 0. resolve \ref / \eqref to real numbers using the .aux label table ----
# (the .aux is written by the LaTeX compile; recompile csar_net.tex before this step)
_aux_path = os.path.join(HERE, 'csar_net.aux')
_labels = {}
if os.path.exists(_aux_path):
    _aux = io.open(_aux_path, encoding='utf-8').read()
    for _m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}', _aux):
        _labels[_m.group(1)] = _m.group(2)
s = re.sub(r'\\eqref\s*\{([^}]+)\}',
           lambda m: '(%s)' % _labels.get(m.group(1), m.group(1)), s)
s = re.sub(r'\\ref\s*\{([^}]+)\}',
           lambda m: _labels.get(m.group(1), m.group(1)), s)

# number float captions so the Word manuscript prints "Table 8:" / "Figure 3:"
# like the PDF (pandoc does not auto-number \caption). The \label is dropped.
def _cap_repl(m):
    body, pfx, name = m.group(1), m.group(2), m.group(3)
    num = _labels.get('%s:%s' % (pfx, name))
    if num is None:
        return m.group(0)
    word = 'Figure' if pfx == 'fig' else 'Table'
    return '\\caption{%s %s: %s}' % (word, num, body)

s = re.sub(r'\\caption\{([^\n]*)\}[ \t]*\n[ \t]*\\label\{(tbl|fig|alg):([a-zA-Z]+)\}',
           _cap_repl, s)

# make the remaining tables pandoc-parseable (pandoc silently drops them):
# unwrap \resizebox around a tabular, drop @{} padding specs and \\[Npt] row spacing.
s = re.sub(r'\\resizebox\{\\textwidth\}\{!\}\{%\s*(?=\\begin\{tabular\})', '', s)
s = s.replace('\\end{tabular}}', '\\end{tabular}')
s = s.replace('@{}', '')
s = s.replace('\\\\[3pt]', '\\\\')

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

# ---- 1.5 render CAS front-matter blocks that pandoc does not recognise ----
# pandoc understands \begin{abstract}, but not the CAS \begin{highlights} /
# \begin{keywords} environments (nor \sep). Without this step the Word
# manuscript shows five unlabelled sentences and a bare keyword blob with a
# stray space before every comma.
def _front_matter(s):
    m = re.search(r'\\begin\{keywords\}(.*?)\\end\{keywords\}', s, re.S)
    if m:
        terms = [re.sub(r'\s+', ' ', t).strip() for t in m.group(1).split(r'\sep')]
        terms = [t for t in terms if t]
        s = s[:m.start()] + '\\textbf{Keywords:} ' + ', '.join(terms) + '\n' + s[m.end():]
    m = re.search(r'\\begin\{highlights\}(.*?)\\end\{highlights\}', s, re.S)
    if m:
        items = re.findall(r'\\item\s*(.*?)(?=\\item|\Z)', m.group(1), re.S)
        items = [re.sub(r'\s+', ' ', t).strip() for t in items if t.strip()]
        bullets = '\n'.join('  \\item ' + t for t in items)
        s = s[:m.start()] + '\\textbf{Highlights}\n\\begin{itemize}\n' + bullets + '\n\\end{itemize}\n' + s[m.end():]
    return s

s = _front_matter(s)

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
