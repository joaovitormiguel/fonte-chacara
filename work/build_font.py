"""Vectorize glyph masks and build Chacara-Regular.otf / .ttf (+ woff2 if possible)."""
import sys, os, json, cv2, numpy as np, zlib
sys.path.insert(0, 'work')
from lib import *
from vec import mask_to_beziers
import registry
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString

FAMILY = 'Chacara'
VERSION = '1.000'
SB = 40
OUT = 'build'
os.makedirs(OUT, exist_ok=True)
meta = json.load(open('work/masks/meta.json'))

def get_mask(nm):
    c = cv2.imread(f'work/masks/{nm}.png', 0) > 127
    x0, y0, x1, y1 = meta[nm]
    return place(c, x0, y0)

# contour smoothing (font units) — originals need more to remove wood-grain noise
SMOOTH = {}  # per-glyph overrides of contour smoothing (font units)

# ---------------------------------------------------------------- outlines
outlines = {}   # name -> (contours, advance)
masks = {}
for nm in meta:
    m = get_mask(nm)
    masks[nm] = m
    cs = mask_to_beziers(m, smooth_units=SMOOTH.get(nm, 8), err=2.0)
    outlines[nm] = cs

def shift(cs, dx, dy=0):
    return [[s + np.array([dx, dy]) for s in segs] for segs in cs]

def ink_bbox(cs):
    p = np.vstack([np.vstack(s) for s in cs]); return p.min(0), p.max(0)

glyphs = {}   # name -> (contours, advance)
for nm, cs in outlines.items():
    if nm in registry.ACCENTS:
        continue
    (x0, y0), (x1, y1) = ink_bbox(cs)
    glyphs[nm] = (shift(cs, SB - x0), int(round(x1 - x0 + 2 * SB)))

# punctuation spacing tweaks
for nm, extra in {'period': 10, 'comma': 10, 'colon': 10, 'semicolon': 10, 'exclam': 10,
                  'quotesingle': 0, 'quotedbl': 0, 'hyphen': 0}.items():
    cs, adv = glyphs[nm]; glyphs[nm] = (shift(cs, extra / 2), adv + extra)

# ---------------------------------------------------------------- accents / composites
def top_center(nm):
    """x of the visual center of the glyph's top part (in final glyph coords)."""
    cs, adv = glyphs[nm]
    m = masks[nm]; x0m = bbox(m)[0]
    (gx0, _), _ = ink_bbox(cs)
    ys, xs = np.nonzero(m)
    fy = YT - ys
    top = fy.max()
    sel = fy > top - 140
    return (xs[sel] - X0).mean() - x0m + gx0, top

def accent_cs(acc):
    cs = outlines[acc]
    (x0, y0), (x1, y1) = ink_bbox(cs)
    return shift(cs, -(x0 + x1) / 2, -y0), (x1 - x0, y1 - y0)

COMPOSE = {
    'Agrave': ('A', 'grave', 0xC0), 'Aacute': ('A', 'acute', 0xC1), 'Acircumflex': ('A', 'circumflex', 0xC2),
    'Atilde': ('A', 'tilde', 0xC3), 'Adieresis': ('A', 'dieresis', 0xC4),
    'Egrave': ('E', 'grave', 0xC8), 'Eacute': ('E', 'acute', 0xC9), 'Ecircumflex': ('E', 'circumflex', 0xCA),
    'Edieresis': ('E', 'dieresis', 0xCB),
    'Igrave': ('I', 'grave', 0xCC), 'Iacute': ('I', 'acute', 0xCD), 'Icircumflex': ('I', 'circumflex', 0xCE),
    'Idieresis': ('I', 'dieresis', 0xCF), 'Ntilde': ('N', 'tilde', 0xD1),
    'Ograve': ('O', 'grave', 0xD2), 'Oacute': ('O', 'acute', 0xD3), 'Ocircumflex': ('O', 'circumflex', 0xD4),
    'Otilde': ('O', 'tilde', 0xD5), 'Odieresis': ('O', 'dieresis', 0xD6),
    'Ugrave': ('U', 'grave', 0xD9), 'Uacute': ('U', 'acute', 0xDA), 'Ucircumflex': ('U', 'circumflex', 0xDB),
    'Udieresis': ('U', 'dieresis', 0xDC), 'Yacute': ('Y', 'acute', 0xDD),
    'Ccedilla': ('C', 'cedilla', 0xC7),
}
ACC_OFFSET = {'acute': (18, 0), 'grave': (-18, 0)}
for nm, (base, acc, cp) in COMPOSE.items():
    bcs, adv = glyphs[base]
    acs, (aw, ah) = accent_cs(acc)
    cx, top = top_center(base)
    if acc == 'cedilla':
        (bx0, by0), (bx1, by1) = ink_bbox(bcs)
        a = shift(acs, (bx0 + bx1) / 2 + 10, 0)
        (ax0, ay0), (ax1, ay1) = ink_bbox(a)
        a = shift(a, 0, -ay1 + 40)  # hang from the baseline
    else:
        dx, dy = ACC_OFFSET.get(acc, (0, 0))
        a = shift(acs, cx + dx, max(top, 700) + 42 + dy)
    glyphs[nm] = (bcs + a, adv)

# spacing accents
for acc in registry.ACCENTS:
    acs, (aw, ah) = accent_cs(acc)
    y = -150 if acc == 'cedilla' else 742
    if acc == 'cedilla':
        (ax0, ay0), (ax1, ay1) = ink_bbox(acs); acs = shift(acs, 0, -ay1 + 40); y = 0
    glyphs[acc] = (shift(acs, aw / 2 + SB, y), int(aw + 2 * SB))

glyphs['space'] = ([], 250)
glyphs['nbspace'] = ([], 250)
glyphs['.notdef'] = ([], 500)

# ---------------------------------------------------------------- cmap
cmap = {0x20: 'space', 0xA0: 'nbspace'}
for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    cmap[ord(c)] = c; cmap[ord(c.lower())] = c
DIG = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
for i, d in enumerate(DIG): cmap[0x30 + i] = d
PUNCT_CP = {'period': 0x2E, 'comma': 0x2C, 'colon': 0x3A, 'semicolon': 0x3B, 'exclam': 0x21, 'question': 0x3F,
            'hyphen': 0x2D, 'endash': 0x2013, 'emdash': 0x2014, 'quotesingle': 0x27, 'quotedbl': 0x22,
            'quoteleft': 0x2018, 'quoteright': 0x2019, 'quotedblleft': 0x201C, 'quotedblright': 0x201D,
            'parenleft': 0x28, 'parenright': 0x29, 'slash': 0x2F, 'plus': 0x2B, 'equal': 0x3D,
            'guillemotleft': 0xAB, 'guillemotright': 0xBB, 'exclamdown': 0xA1, 'questiondown': 0xBF,
            'dollar': 0x24, 'ordmasculine': 0xBA, 'ordfeminine': 0xAA, 'asterisk': 0x2A,
            'degree': 0xB0, 'percent': 0x25, 'numbersign': 0x23, 'underscore': 0x5F, 'bullet': 0x2022, 'periodcentered': 0xB7,
            'ellipsis': 0x2026, 'ampersand': 0x26, 'acute': 0xB4, 'grave': 0x60, 'tilde': 0x2DC, 'circumflex': 0x2C6, 'dieresis': 0xA8, 'cedilla': 0xB8}
cmap.update({cp: nm for nm, cp in PUNCT_CP.items()})
cmap[0x7E] = 'tilde'; cmap[0x5E] = 'circumflex'; cmap[0x2010] = 'hyphen'; cmap[0x2212] = 'hyphen'
for nm, (b, a, cp) in COMPOSE.items():
    cmap[cp] = nm; cmap[cp + 0x20] = nm   # lowercase accented -> same glyph

order = ['.notdef', 'space', 'nbspace'] + sorted(n for n in glyphs if n not in ('.notdef', 'space', 'nbspace'))

# ---------------------------------------------------------------- features: contextual alternates
def base_of(g):
    return g.split('.')[0]
def var_idx(g):
    return int(g.split('.alt')[1]) if '.alt' in g else 0
letter_like = [g for g in order if base_of(g) in registry.LETTERS or g in COMPOSE or g in DIG]
fea = ['languagesystem DFLT dflt;', 'languagesystem latn dflt;', '']
# stylistic sets: pick a specific version of every letter (ss01 = 2nd version, ss02 = 3rd)
for k, label in [(1, 'Segunda versão das letras'), (2, 'Terceira versão das letras')]:
    subs = [f'  sub {X} by {X}.alt{k};' for X, r in registry.LETTERS.items() if len(r) > k]
    fea += [f'feature ss0{k} {{', f'  featureNames {{ name "{label}"; }};'] + subs + [f'}} ss0{k};', '']
fea.append('feature calt {')
for X, recipes in registry.LETTERS.items():
    n = len(recipes)
    if n < 2: continue
    groups = {k: [] for k in range(1, n)}
    for g in letter_like:
        if base_of(g) == X:
            k = (var_idx(g) + 1) % n
        else:
            k = (zlib.crc32((base_of(g) + '>' + X).encode()) + var_idx(g)) % n
        if k: groups[k].append(g)
    for k, gl in groups.items():
        if gl:
            fea.append(f"  sub [{' '.join(gl)}] {X}' by {X}.alt{k};")
fea.append('} calt;')

# ---------------------------------------------------------------- kerning (profile based)
def profiles(nm, r=70, step=10):
    cs, adv = glyphs[nm]
    ys = np.arange(-300, 950, step)
    L = np.full(len(ys), np.nan); R = np.full(len(ys), np.nan)
    if not cs: return ys, L, R, adv
    from vec import bez
    t = np.linspace(0, 1, 24)
    pts = np.vstack([np.vstack([bez(s, t) for s in segs]) for segs in cs])
    for i, y in enumerate(ys):
        sel = np.abs(pts[:, 1] - y) < step / 2 + 0.5
        if sel.any():
            L[i] = pts[sel, 0].min(); R[i] = pts[sel, 0].max()
    # circular dilation of the profiles
    Ld = np.full(len(ys), np.inf); Rd = np.full(len(ys), -np.inf)
    k = int(r / step)
    for dk in range(-k, k + 1):
        h = np.sqrt(max(r * r - (dk * step) ** 2, 0))
        Ls = np.roll(L, dk); Rs = np.roll(R, dk)
        if dk > 0: Ls[:dk] = np.nan; Rs[:dk] = np.nan
        if dk < 0: Ls[dk:] = np.nan; Rs[dk:] = np.nan
        Ld = np.fmin(Ld, Ls - h); Rd = np.fmax(Rd, Rs + h)
    Ld[np.isinf(Ld)] = np.nan; Rd[np.isinf(Rd)] = np.nan
    return ys, Ld, Rd, adv

kern_glyphs = list(registry.LETTERS) + DIG + ['period', 'comma', 'hyphen', 'quoteright', 'quotedblright',
                                               'quoteleft', 'quotedblleft', 'colon', 'semicolon']
P = {g: profiles(g) for g in kern_glyphs}
def gap(a, b):
    ys, La, Ra, adva = P[a]; _, Lb, Rb, _ = P[b]
    d = (adva + Lb) - Ra
    return np.nanmin(d) if np.isfinite(d).any() else None
ref = np.mean([gap(a, b) for a in 'HIN' for b in 'HIN'])
kerns = {}
for a in kern_glyphs:
    for b in kern_glyphs:
        g = gap(a, b)
        if g is None: continue
        k = ref - g
        if k < 0: k = max(-170, 0.6 * k)
        else: k = min(90, k)
        k = int(round(k / 5) * 5)
        if abs(k) >= 20:
            kerns[(a, b)] = k
classes = {}
for g in order:
    b = base_of(g)
    if g in COMPOSE: b = COMPOSE[g][0]
    if b in kern_glyphs:
        classes.setdefault(b, []).append(g)
fea.append('')
for b, gl in classes.items():
    fea.append(f"@K_{b} = [{' '.join(gl)}];")
fea.append('feature kern {')
for (a, b), k in sorted(kerns.items()):
    fea.append(f'  pos @K_{a} @K_{b} {k};')
fea.append('} kern;')
FEA = '\n'.join(fea)
open('work/features.fea', 'w').write(FEA)
print('kern pairs:', len(kerns), ' ref gap:', round(ref, 1))

# ---------------------------------------------------------------- build
def draw(pen, cs):
    for segs in cs:
        p0 = segs[0][0]
        pen.moveTo((round(p0[0]), round(p0[1])))
        for s in segs:
            pen.curveTo(*[(round(x), round(y)) for x, y in s[1:]])
        pen.closePath()

names = dict(familyName=FAMILY, styleName='Regular', uniqueFontIdentifier=f'{FAMILY}-Regular-{VERSION}',
             fullName=f'{FAMILY} Regular', psName=f'{FAMILY}-Regular', version=f'Version {VERSION}',
             copyright='Lettering: placa da Chácara São Francisco',
             description='Fonte baseada nas letras de madeira recortadas à mão da placa da Chácara São Francisco.')
metrics = {n: (adv, 0) for n, (cs, adv) in glyphs.items()}

def common(fb):
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupHorizontalMetrics({n: (glyphs[n][1], 0) for n in order})
    fb.setupHorizontalHeader(ascent=920, descent=-260)
    fb.setupNameTable(names)
    fb.setupOS2(sTypoAscender=920, sTypoDescender=-260, sTypoLineGap=60, usWinAscent=1000, usWinDescent=320,
                sxHeight=700, sCapHeight=700, achVendID='CHSF', fsType=0, usWeightClass=400, fsSelection=0x40)
    fb.setupPost()

def fix_lsb(font):
    # recompute LSBs from real bounds
    hmtx = font['hmtx']
    gs = font.getGlyphSet()
    from fontTools.pens.boundsPen import BoundsPen
    for n in order:
        bp = BoundsPen(gs); gs[n].draw(bp)
        hmtx[n] = (hmtx[n][0], int(bp.bounds[0]) if bp.bounds else 0)

# OTF (CFF)
fb = FontBuilder(1000, isTTF=False)
common(fb)
cs_map = {}
for n in order:
    pen = T2CharStringPen(glyphs[n][1], None)
    draw(pen, glyphs[n][0])
    cs_map[n] = pen.getCharString()
fb.setupCFF(names['psName'], {'FullName': names['fullName'], 'FamilyName': FAMILY, 'Weight': 'Regular'}, cs_map, {})
addOpenTypeFeaturesFromString(fb.font, FEA)
fix_lsb(fb.font)
fb.save(f'{OUT}/{FAMILY}-Regular.otf')

# TTF
fb = FontBuilder(1000, isTTF=True)
common(fb)
tt = {}
for n in order:
    pen = TTGlyphPen(None)
    draw(Cu2QuPen(pen, 1.0, reverse_direction=True), glyphs[n][0])
    tt[n] = pen.glyph()
fb.setupGlyf(tt)
fb.setupMaxp()
addOpenTypeFeaturesFromString(fb.font, FEA)
fix_lsb(fb.font)
fb.save(f'{OUT}/{FAMILY}-Regular.ttf')
try:
    from fontTools.ttLib import TTFont
    f = TTFont(f'{OUT}/{FAMILY}-Regular.ttf'); f.flavor = 'woff2'; f.save(f'{OUT}/{FAMILY}-Regular.woff2')
except Exception as e:
    print('woff2 skipped:', e)
print('glyphs:', len(order))
