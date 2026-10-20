"""Glyph recipes. Originals come from the sign photo; the rest are built in the same style
(very heavy rounded strokes, counters reduced to small holes/slits, hand-cut wobble)."""
import cv2, numpy as np
from scipy import ndimage as ndi
from lib import *

def load(name):
    return cv2.imread(f'work/norm/{name}.png', 0) > 127

def spline(pts, n=12):
    """Catmull-Rom through pts (open)."""
    p = np.asarray(pts, float)
    p = np.vstack([p[0] * 2 - p[1], p, p[-1] * 2 - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-2])
    return np.array(out)

def fill_holes(m):
    return ndi.binary_fill_holes(m)

def hole(cx, cy, r=27, ry=None):
    return ellipse(cx, cy, r, ry if ry else r * 1.08)

SW = 172  # stem weight measured on the sign (~150-180)

# ---------------------------------------------------------------- originals
CLEAN = {'H': (22, 19), 'F': (20, 17), 'R2': (18, 15), 'N': (17, 14), 'C2': (18, 15), 'C3': (16, 13), 'C4': (16, 13)}
FAIR = {'H': 36, 'F': 30, 'R2': 28, 'N': 28, 'C2': 34, 'C3': 30, 'C4': 30, 'S1': 24, 'S2': 24, 'I': 26}
SQUEEZE = {'S1': 0.86}
BLUR = {'H': 15, 'F': 15, 'R2': 14, 'N': 13, 'C2': 14, 'C3': 13, 'C4': 13, 'A3': 13, 'A5': 12}
def orig(name, flip=False):
    def f(seed_name):
        m = place(load(name), 0, 0, flipx=flip)
        ci, co = CLEAN.get(name, (13, 11))
        m = soften_inner(m, ci)          # fill small dents left by wood grain
        m = round_corners(m, co)         # remove small bumps
        m = blur_thresh(m, BLUR.get(name, 11))
        m = polish(m, FAIR.get(name, 20))  # fair the outline, clean round counters
        if name in SQUEEZE:
            x0, y0, x1, y1 = bbox(m)
            m = place(crop(m), 0, 0, sx=SQUEEZE[name])
        return m
    f.original = True
    return f

def overshoot(m, ov=12):
    """Scale a 0..700 glyph to -ov..700+ov (round letters)."""
    x0, y0, x1, y1 = bbox(m)
    c = crop(m)
    s = (CAP + 2 * ov) / c.shape[0]
    return place(c, x0, -ov, sx=s, sy=s)

# ---------------------------------------------------------------- constructed letters
def B(n):
    m = rect(0, 0, 175, 700) | rect(0, 0, 270, 700) & (FX < 270)
    m |= ellipse(262, 528, 196, 178) | ellipse(268, 205, 212, 208)
    m = finish(m, n, corner=48)
    return m & ~hole(238, 532, 27) & ~hole(245, 205, 31)

def D(n):
    m = rect(0, 0, 250, 700) | ellipse(252, 350, 250, 352)
    m = finish(m, n, corner=55)
    return m & ~hole(232, 352, 27, 62)

def E(n):
    m = rect(0, 0, 176, 700)
    m |= stroke([(80, 612), (395, 612)], 176) | rect(0, 612, 395, 700)
    m |= stroke([(80, 355), (330, 362)], 168)
    m |= stroke([(80, 88), (410, 86)], 176) | rect(0, 0, 410, 88)
    return finish(m, n, corner=46)

def G(n):
    c = load('C3')
    m = place(c, 0, 0)
    x0, y0, x1, y1 = bbox(m)
    m |= stroke([(x1 - 230, 318), (x1 - 75, 318), (x1 - 80, 130)], 158)
    return overshoot(finish(m, n, clip=None, corner=20, amp=4))

def J(n):
    pts = np.vstack([[(318, 790), (318, 250)], arc(190, 222, 128, 135, 0, -175)])
    m = stroke(pts, SW)
    return finish(m, n, corner=30)

def K(n):
    m = rect(0, 0, 176, 700)
    m |= stroke([(140, 330), (430, 790)], 176)
    m |= stroke([(165, 395), (440, -60)], 182)
    return finish(m, n, corner=40)

def L(n):
    m = rect(0, 0, 178, 700) | stroke([(88, 88), (405, 88)], 178) | rect(0, 0, 405, 88)
    return finish(m, n, corner=46)

def M(n):
    m = rect(0, 0, 172, 700) | rect(458, 0, 630, 700)
    m |= stroke([(60, 800), (315, 95), (570, 800)], 180)
    return finish(m, n, corner=40)

def P(n):
    m = rect(0, 0, 176, 700) | (rect(0, 250, 250, 700)) | ellipse(250, 480, 215, 222)
    m = finish(m, n, corner=48)
    return m & ~hole(232, 488, 27)

def Q(n):
    m = overshoot(place(load('O1'), 0, 0))
    m |= stroke([(330, 130), (540, -95)], 150, 120)
    return finish(m, n, clip=(-110, 760), corner=20, amp=3)

def T(n):
    m = stroke([(80, 612), (440, 612)], 178) | rect(80, 612, 440, 700)
    m |= rect(170, 0, 350, 650)
    return finish(m, n, corner=46)

def U(n):
    m = rect(0, 230, 452, 700) | ellipse(226, 230, 226, 232)
    m &= ~capsule_chain(dense([(226, 820), (226, 262)]), 58, 70)[::1]
    return finish(m, n, corner=40)

def V(n):
    m = stroke([(80, 790), (250, 100), (420, 790)], 190) | ellipse(250, 118, 118, 118)
    return finish(m, n, corner=40)

def W(n):
    m = stroke([(84, 780), (150, -80)], 182) | stroke([(636, 780), (570, -80)], 182)
    m |= stroke([(165, 30), (360, 610), (555, 30)], 178)
    return finish(m, n, corner=40)

def X(n):
    m = stroke([(70, 790), (450, -90)], 190) | stroke([(450, 790), (70, -90)], 190)
    return finish(m, n, corner=40)

def Y(n):
    m = stroke([(60, 800), (255, 330)], 186) | stroke([(450, 800), (255, 330)], 186)
    m |= rect(165, 0, 345, 360)
    return finish(m, n, corner=40)

def Z(n):
    m = stroke([(80, 612), (390, 612)], 178) | rect(0, 612, 390, 700)
    m |= stroke([(395, 590), (95, 110)], 192)
    m |= stroke([(80, 88), (410, 88)], 178) | rect(70, 0, 410, 88)
    return finish(m, n, corner=42)

# ---------------------------------------------------------------- digits
def zero(n):
    o = fill_holes(load('O1'))
    m = place(o, 0, 0, sx=0.84)
    x0, y0, x1, y1 = bbox(m)
    m = overshoot(m)
    return m & ~ellipse((x0 + x1) / 2, 352, 28, 92)

def one(n):
    m = place(load('I'), 120, 0)
    m |= stroke([(175, 600), (40, 500)], 158, 140)
    return finish(m, n, clip=None, corner=20, amp=3)

def two(n):
    pts = np.vstack([arc(228, 478, 150, 140, 168, -28), [(85, 88), (420, 88)]])
    m = stroke(pts, SW) | rect(0, 0, 420, 88)
    return finish(m, n, clip=(0, 760), corner=40)

def three(n):
    pts = np.vstack([arc(215, 528, 150, 108, 165, -90), arc(215, 214, 168, 126, 90, -165)])
    m = stroke(pts, SW)
    return overshoot(finish(m, n, clip=None, corner=30))

def four(n):
    m = stroke([(320, 680), (78, 262)], 176) | stroke([(40, 220), (470, 220)], 168)
    m |= rect(250, 0, 420, 700)
    m = finish(m, n, corner=40)
    return m & ~hole(228, 338, 27, 34)

def five(n):
    pts = np.vstack([[(420, 612), (118, 612), (110, 412)], arc(240, 250, 172, 160, 110, -140)])
    m = stroke(pts, SW) | rect(30, 612, 420, 700)
    return finish(m, n, clip=(-12, 700), corner=34)

def six(n):
    m = ellipse(240, 232, 235, 236)
    m |= stroke(spline([(95, 230), (110, 450), (200, 600), (340, 630), (410, 610)]), SW)
    m = finish(m, n, clip=(-12, 712), corner=30)
    return m & ~hole(240, 228, 29)

def nine(n):
    m = six(n + 'x')
    return place(crop(m)[::-1, ::-1], 0, -12)

def seven(n):
    m = stroke([(60, 612), (430, 612)], 178) | rect(60, 612, 430, 700)
    m |= stroke([(400, 600), (170, -60)], 186)
    return finish(m, n, corner=42)

def eight(n):
    m = ellipse(235, 532, 178, 180) | ellipse(235, 212, 222, 222)
    m = finish(m, n, clip=None, corner=0)
    m = overshoot(m, 10)
    return m & ~hole(235, 535, 26) & ~hole(235, 210, 30)

# ---------------------------------------------------------------- punctuation
def dot_at(cx, cy, r=82):
    return ellipse(cx, cy, r * 1.04, r)

def period(n):   return finish(dot_at(95, 84), n, clip=None, corner=0, amp=4)
def comma(n):
    m = dot_at(100, 92, 84) | stroke([(140, 60), (60, -150)], 110, 70)
    return finish(m, n, clip=None, corner=0, amp=4)
def colon(n):    return finish(dot_at(95, 84) | dot_at(95, 440), n, clip=None, corner=0, amp=4)
def semicolon(n):
    m = dot_at(100, 440, 84) | dot_at(100, 92, 84) | stroke([(140, 60), (60, -150)], 110, 70)
    return finish(m, n, clip=None, corner=0, amp=4)
def exclam(n):
    m = stroke([(100, 790), (98, 290)], 186, 140) | dot_at(98, 80, 80)
    return finish(m, n, clip=(-10, 700), corner=30, amp=5)
def question(n):
    pts = np.vstack([arc(232, 505, 150, 128, 170, -62), [(268, 270)]])
    m = stroke(pts, 168) | dot_at(262, 80, 80)
    return finish(m, n, clip=(-10, 760), corner=30, amp=5)
def hyphen(n):   return finish(stroke([(70, 300), (330, 300)], 150), n, clip=None, corner=0, amp=5)
def endash(n):   return finish(stroke([(70, 300), (500, 300)], 150), n, clip=None, corner=0, amp=5)
def emdash(n):   return finish(stroke([(70, 300), (930, 300)], 150), n, clip=None, corner=0, amp=5)
def quotesingle(n): return finish(stroke([(80, 690), (78, 500)], 135, 100), n, clip=None, corner=0, amp=4)
def quotedbl(n):
    return finish(stroke([(80, 690), (78, 500)], 135, 100) | stroke([(250, 690), (248, 500)], 135, 100), n, clip=None, corner=0, amp=4)
def _commaish(x, y):
    return dot_at(x, y, 70) | stroke([(x + 35, y - 25), (x - 30, y - 165)], 92, 55)
def quoteright(n): return finish(_commaish(100, 620), n, clip=None, corner=0, amp=3)
def quotedblright(n): return finish(_commaish(100, 620) | _commaish(270, 620), n, clip=None, corner=0, amp=3)
def quoteleft(n):
    m = quoteright(n + 'x'); return place(crop(m)[::-1, ::-1], 20, 470)
def quotedblleft(n):
    m = quotedblright(n + 'x'); return place(crop(m)[::-1, ::-1], 20, 470)
def parenleft(n):
    m = stroke(arc(390, 340, 290, 470, 128, 232), 150)
    return finish(m, n, clip=None, corner=0, amp=5)
def parenright(n):
    m = parenleft(n + 'x'); c = crop(m); x0, y0, x1, y1 = bbox(m)
    return place(c[:, ::-1], 0, y0)
def slash(n):    return finish(stroke([(60, -60), (360, 760)], 152), n, clip=(-80, 780), corner=0, amp=5)
def plus(n):
    return finish(stroke([(60, 320), (420, 320)], 150) | stroke([(240, 140), (240, 500)], 150), n, clip=None, corner=0, amp=5)
def equal(n):
    return finish(stroke([(60, 420), (400, 420)], 140) | stroke([(60, 200), (400, 200)], 140), n, clip=None, corner=0, amp=5)
def guillemotleft(n):
    a = stroke([(250, 480), (90, 300), (250, 120)], 130) | stroke([(450, 480), (290, 300), (450, 120)], 130)
    return finish(a, n, clip=None, corner=0, amp=4)
def guillemotright(n):
    m = guillemotleft(n + 'x'); x0, y0, x1, y1 = bbox(m)
    return place(crop(m)[:, ::-1], 0, y0)
def exclamdown(n):
    m = exclam(n + 'x'); return place(crop(m)[::-1, ::-1], 0, -230)
def questiondown(n):
    m = question(n + 'x'); return place(crop(m)[::-1, ::-1], 0, -230)
def dollar(n):
    m = place(load('S2'), 0, 0)
    x0, y0, x1, y1 = bbox(m)
    cx = (x0 + x1) / 2
    m |= stroke([(cx, 600), (cx, 800)], 92) | stroke([(cx, 100), (cx, -100)], 92)
    return finish(m, n, clip=None, corner=0, amp=3)
def ordmasculine(n):
    m = place(load('O1'), 0, 0, sx=0.42, sy=0.42)
    m = place(crop(m), 0, 420)
    return m
def ordfeminine(n):
    m = place(load('A2'), 0, 0, sx=0.42, sy=0.42)
    return place(crop(m), 0, 420)
def asterisk(n):
    m = np.zeros_like(blank())
    for a in (90, 162, 234, 306, 18):
        t = np.radians(a)
        m |= stroke([(200, 480), (200 + 140 * np.cos(t), 480 + 140 * np.sin(t))], 92)
    return finish(m, n, clip=None, corner=0, amp=4)

# ---------------------------------------------------------------- accents (y placed later)
def acute(n):
    m = ellipse(92, 62, 96, 56, angle=38) | ellipse(66, 46, 64, 50)
    return finish(m, n, clip=None, corner=0, amp=3, scale=30)
def grave(n):
    m = acute(n + 'x'); return place(crop(m)[:, ::-1], 0, 0)
def tilde(n):
    m = stroke(spline([(0, 16), (65, 62), (145, 46), (215, 32), (285, 62)], 10), 92, 98)
    return finish(m, n, clip=None, corner=0, amp=3, scale=30)
def circumflex(n):
    m = stroke([(0, 0), (115, 100), (230, 0)], 96)
    return finish(m, n, clip=None, corner=0, amp=3)
def dieresis(n):
    m = ellipse(56, 56, 60, 56) | ellipse(226, 56, 60, 56)
    return finish(m, n, clip=None, corner=0, amp=3)
def cedilla(n):
    pts = spline([(60, 40), (72, -45), (140, -80), (128, -150), (30, -180)], 10)
    m = stroke(pts, 96, 80)
    return finish(m, n, clip=None, corner=0, amp=2)

# ---------------------------------------------------------------- extra symbols
def degree(n):
    m = ellipse(120, 590, 112, 112)
    m = finish(m, n, clip=None, corner=0, amp=4)
    return m & ~hole(120, 590, 40)
def percent(n):
    m = ellipse(120, 545, 118, 150) | ellipse(470, 155, 118, 150)
    m |= stroke([(150, 0), (440, 700)], 120)
    m = finish(m, n, clip=None, corner=0, amp=5)
    return m & ~hole(120, 548, 26, 40) & ~hole(470, 158, 26, 40)
def numbersign(n):
    m = stroke([(170, 40), (230, 660)], 112) | stroke([(370, 40), (430, 660)], 112)
    m |= stroke([(40, 470), (560, 470)], 108) | stroke([(25, 230), (545, 230)], 108)
    return finish(m, n, clip=None, corner=0, amp=5)
def underscore(n): return finish(stroke([(60, -90), (480, -90)], 110), n, clip=None, corner=0, amp=4)
def bullet(n):     return finish(dot_at(110, 330, 100), n, clip=None, corner=0, amp=4)
def ellipsis(n):
    return finish(dot_at(95, 84) | dot_at(305, 84) | dot_at(515, 84), n, clip=None, corner=0, amp=4)
def ampersand(n):
    m = ellipse(225, 545, 150, 155) | ellipse(250, 225, 225, 225)
    m |= stroke([(250, 330), (560, -20)], 175)
    m |= stroke([(330, 330), (500, 380)], 150)
    m = finish(m, n, corner=30, clip=(0, 700))
    return m & ~hole(222, 552, 25) & ~hole(215, 215, 30)
def periodcentered(n): return finish(dot_at(95, 330, 72), n, clip=None, corner=0, amp=4)
