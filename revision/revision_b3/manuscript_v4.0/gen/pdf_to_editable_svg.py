"""Convert a vector PDF figure (here the TikZ figures) into an SVG whose shapes are paths and whose text is
editable <text>.  Shapes and glyph positions come from poppler's pdftocairo; the text content, font and
colour of each run come from pdftohtml.  Glyphs whose text cannot be recovered (e.g. big delimiters) are
kept as outlines, so nothing is lost.

python pdf_to_editable_svg.py in.pdf out.svg
"""
import re, sys, html, subprocess, tempfile, pathlib, statistics
from collections import Counter

SANS = "Helvetica, Arial, 'TeX Gyre Heros', 'Nimbus Sans', sans-serif"
SERIF = "'Times New Roman', Times, 'TeX Gyre Termes', serif"
MATH = "'Cambria Math', 'STIX Two Math', 'Times New Roman', 'TeX Gyre Termes', serif"
MATH_AS_TEXT = False      # mathematics stays as exact outlines; all prose labels become editable text
MAP = {('CMEX', 'P'): '∑', ('MSBM', 'R'): 'ℝ', ('CMSY', 'N'): '\U0001d4a9', ('CMSY', 'L'): 'ℒ'}


def style_for(family):
    fam = family.split('+')[-1]
    if fam.startswith('NimbusSanL') or 'Helvetica' in fam:
        return fam, SANS, 'bold' if 'Bold' in fam else 'normal', 'normal'
    if fam.startswith('CMMI'):
        return fam, SERIF, 'normal', 'italic'
    if fam.startswith('CMBX'):
        return fam, SERIF, 'bold', 'normal'
    if fam.startswith(('CMSY', 'CMEX', 'MSBM', 'MSAM')):
        return fam, MATH, 'normal', 'normal'
    return fam, SERIF, 'normal', 'normal'


HEROS = {'normal': '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf',
         'bold': '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf'}


def calibrate_sizes(runs):
    """pdftohtml rounds font sizes to whole units, which makes long labels up to ~2% too wide.  Recover the exact
    size of each sans font from the distance between the first and last glyph of its runs, measured against
    Helvetica metrics (TeX Gyre Heros), and use the median over the runs of that font."""
    from PIL import ImageFont
    est = {}
    for r in runs:
        fam, ff, fw, _ = style_for(r['family'])
        t = r['text'].strip()
        g = sorted(r['uses'], key=lambda u: u['x'])
        if ff != SANS or len(t.replace(' ', '')) != len(g) or len(g) < 6:
            continue                                    # ligatures or too short to measure
        font = ImageFont.truetype(HEROS[fw], 1000)
        adv = font.getlength(t) - font.getlength(t[-1])
        if adv > 0:
            est.setdefault(r['fid'], []).append((g[-1]['x'] - g[0]['x']) / adv * 1000)
    for r in runs:
        if r['fid'] in est:
            r['size'] = statistics.median(est[r['fid']])


def main(pdf, out):
    tmp = pathlib.Path(tempfile.mkdtemp())
    subprocess.run(['pdftocairo', '-svg', pdf, str(tmp / 'c.svg')], check=True)
    subprocess.run(['pdftohtml', '-xml', '-i', '-q', '-fontfullname', '-zoom', '4', pdf, str(tmp / 'h')], check=True)
    svg = (tmp / 'c.svg').read_text(encoding='utf-8')
    xml = (tmp / 'h.xml').read_text(encoding='utf-8')
    W = float(re.search(r'<svg[^>]*\swidth="([\d.]+)', svg).group(1))
    PW = float(re.search(r'<page [^>]*width="([\d.]+)"', xml).group(1))
    s = W / PW
    fonts = {m.group(1): (float(m.group(2)) * s, m.group(3), m.group(4))
             for m in re.finditer(r'<fontspec id="(\d+)" size="([\d.]+)" family="([^"]+)" color="([^"]+)"', xml)}
    # glyph placements: (index in svg, font face index, x, y)
    uses = [dict(span=m.span(), face=m.group(1), x=float(m.group(3)), y=float(m.group(4)), run=None)
            for m in re.finditer(r'<use xlink:href="#glyph-(\d+)-(\d+)" x="([-\d.]+)" y="([-\d.]+)"\s*/>', svg)]
    runs = []
    for m in re.finditer(r'<text top="([-\d.]+)" left="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)" font="(\d+)">(.*?)</text>', xml, re.S):
        T, L, Wd, H = (float(m.group(k)) * s for k in range(1, 5))
        size, family, color = fonts[m.group(5)]
        text = html.unescape(re.sub(r'<[^>]+>', '', m.group(6)))
        runs.append(dict(T=T, L=L, W=Wd, H=H, size=size, family=family, color=color, text=text, uses=[], fid=m.group(5)))
    # assign glyphs to runs: candidates inside the run box, restricted to the run's dominant font face
    for r in runs:
        cand = [u for u in uses if u['run'] is None and r['L'] - 1.0 <= u['x'] <= r['L'] + r['W'] + 0.5 and r['T'] - 1.0 <= u['y'] <= r['T'] + r['H'] + 2.5]
        if not cand:
            continue
        face = Counter(u['face'] for u in cand).most_common(1)[0][0]
        for u in cand:
            if u['face'] == face:
                u['run'] = id(r); r['uses'].append(u)
    calibrate_sizes(runs)
    texts, drop = [], set()
    for r in runs:
        fam, ff, fw, fs = style_for(r['family'])
        t = r['text']
        key = next((k for k in MAP if fam.startswith(k[0]) and t == k[1]), None)
        if key:
            t = MAP[key]
        if not t.strip() or not r['uses'] or (ff != SANS and not MATH_AS_TEXT):
            continue                                   # keep the outlines (mathematics, or nothing to recover)
        x = min(u['x'] for u in r['uses']); y = statistics.median(u['y'] for u in r['uses'])
        for u in r['uses']:
            drop.add(u['span'])
        g = sorted(r['uses'], key=lambda u: u['x'])
        compact = t.replace(' ', '')
        if ff != SANS and len(compact) == len(g):
            # mathematics: place every character where TeX placed its glyph
            t = compact
            xs = ' '.join('%.3f' % u['x'] for u in g); ys = ' '.join('%.3f' % u['y'] for u in g)
            attrs = 'x="%s" y="%s" font-family="%s" font-size="%.2f" fill="%s"' % (xs, ys, ff.replace('"', "'"), r['size'], r['color'])
        else:
            attrs = 'x="%.3f" y="%.3f" font-family="%s" font-size="%.2f" fill="%s"' % (x, y, ff.replace('"', "'"), r['size'], r['color'])
        if fw != 'normal':
            attrs += ' font-weight="%s"' % fw
        if fs != 'normal':
            attrs += ' font-style="%s"' % fs
        texts.append('<text xml:space="preserve" %s>%s</text>' % (attrs, html.escape(t, quote=False)))
    # remove the glyph placements that became text (from the end, so spans stay valid)
    for a, b in sorted(drop, reverse=True):
        svg = svg[:a] + svg[b:]
    svg = svg.replace('</svg>', '<g id="editable-text">\n' + '\n'.join(texts) + '\n</g>\n</svg>')
    pathlib.Path(out).write_text(svg, encoding='utf-8')
    kept = sum(1 for u in uses if u['span'] not in drop)
    print('%s: %d text runs as <text>, %d glyphs kept as outlines' % (out, len(texts), kept))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
