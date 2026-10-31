# -*- coding: utf-8 -*-
"""Restyle pandoc's default reference.docx into a clean academic Word template:
   Times New Roman everywhere, black bold headings, a 16pt bold centered title,
   12pt body/abstract, justified body text, and bordered tables with a shaded
   header row."""
import zipfile, re, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '_custom-reference.docx')

zin = zipfile.ZipFile(SRC)
styles = zin.read('word/styles.xml').decode('utf-8')

TNR = 'Times New Roman'
EA = '宋体'  # SimSun (宋体), for any stray CJK text

def rf():
    return ('<w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s" />'
            % (TNR, TNR, EA, TNR))


def set_style_rpr(styles, style_id, new_rpr):
    """Replace the <w:rPr>…</w:rPr> inside the paragraph style `style_id`."""
    pat = re.compile(
        r'(<w:style [^>]*w:styleId="%s"[^>]*>.*?)<w:rPr>.*?</w:rPr>(.*?</w:style>)'
        % style_id, re.S)
    out, n = pat.subn(lambda m: m.group(1) + new_rpr + m.group(2), styles, count=1)
    assert n == 1, 'rPr replacement failed for %s (n=%d)' % (style_id, n)
    return out


def set_style_ppr(styles, style_id, new_ppr):
    """Replace the <w:pPr>…</w:pPr> inside the paragraph style `style_id`."""
    pat = re.compile(
        r'(<w:style [^>]*w:styleId="%s"[^>]*>.*?)<w:pPr>.*?</w:pPr>(.*?</w:style>)'
        % style_id, re.S)
    out, n = pat.subn(lambda m: m.group(1) + new_ppr + m.group(2), styles, count=1)
    assert n == 1, 'pPr replacement failed for %s (n=%d)' % (style_id, n)
    return out


# 1. document default font -> Times New Roman (was Calibri via minorHAnsi theme)
styles = styles.replace(
    '<w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia" '
    'w:hAnsiTheme="minorHAnsi" w:cstheme="minorBidi" />',
    rf())

# 2. Title: bold, 16pt, Times New Roman (was 28pt Cambria via majorHAnsi)
styles = set_style_rpr(
    styles, 'Title',
    '<w:rPr>%s<w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>' % rf())

# 3. Author: 12pt, explicitly NOT bold (inherits Title's centering + TNR)
styles = set_style_rpr(
    styles, 'Author',
    '<w:rPr><w:b w:val="0"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>')

# 4. Abstract heading: bold, 12pt, centered (was 10pt)
styles = set_style_rpr(
    styles, 'AbstractTitle',
    '<w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>')

# 5. Abstract body: 12pt, justified (was 10pt, left)
styles = set_style_rpr(
    styles, 'Abstract',
    '<w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>')
styles = set_style_ppr(
    styles, 'Abstract',
    '<w:pPr><w:keepNext/><w:keepLines/>'
    '<w:spacing w:before="100" w:after="300"/><w:jc w:val="both"/></w:pPr>')

# 6. Heading 1: black, bold, 14pt, TNR (was blue 20pt)
styles = set_style_rpr(
    styles, 'Heading1',
    '<w:rPr>%s<w:b/><w:color w:val="000000"/>'
    '<w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>' % rf())

# 7. Heading 2: black, bold, 12pt, TNR (was blue 16pt)
styles = set_style_rpr(
    styles, 'Heading2',
    '<w:rPr>%s<w:b/><w:color w:val="000000"/>'
    '<w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>' % rf())

# 8. Body text: justified, 6pt after, 1.15 line spacing (was left, 9pt gaps)
styles = set_style_ppr(
    styles, 'BodyText',
    '<w:pPr><w:spacing w:before="0" w:after="120" '
    'w:line="276" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>')

# 9. Tables: full black borders, cell padding, shaded bold header row
old_table = re.compile(
    r'<w:style w:type="table"[^>]*w:styleId="Table".*?</w:style>', re.S)
new_table = (
    '<w:style w:type="table" w:default="1" w:styleId="Table">'
    '<w:name w:val="Table"/><w:basedOn w:val="TableNormal"/><w:qFormat/>'
    '<w:tblPr>'
    '<w:tblBorders>'
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '</w:tblBorders>'
    '<w:tblCellMar>'
    '<w:top w:w="60" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>'
    '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="108" w:type="dxa"/>'
    '</w:tblCellMar>'
    '</w:tblPr>'
    '<w:tblStylePr w:type="firstRow">'
    '<w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>'
    '<w:vAlign w:val="center"/></w:tcPr>'
    '</w:tblStylePr>'
    '</w:style>')
styles, nt = old_table.subn(new_table, styles, count=1)
assert nt == 1, 'Table style replacement failed (n=%d)' % nt

# write back into the zip
import shutil
tmp = SRC + '.tmp'
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name in zin.namelist():
        data = zin.read(name)
        if name == 'word/styles.xml':
            data = styles.encode('utf-8')
        zout.writestr(name, data)
zin.close()
shutil.move(tmp, SRC)
print('restyled', SRC)
