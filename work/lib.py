"""Raster glyph construction toolkit. 1 px = 1 font unit. y is up (font coords)."""
import cv2, numpy as np, hashlib
from scipy import ndimage as ndi

CAP = 700
X0 = 300          # canvas column of x=0
YT = 1100         # font y at canvas row 0
W, H = 1600, 1600 # canvas size -> x in [-300,1300], y in [-500,1100]

def blank():
    return np.zeros((H, W), bool)

_yy, _xx = np.mgrid[0:H, 0:W]
FX = (_xx - X0).astype(np.float32)   # font x of each pixel
FY = (YT - _yy).astype(np.float32)   # font y of each pixel

def seed_of(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

# ---------- primitives (all return bool canvas) ----------
def capsule_chain(pts, w, w_end=None):
    """Thick polyline with round joins/caps. pts in font units. width may taper linearly to w_end."""
    pts = np.asarray(pts, np.float32)
    out = blank()
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg_len.sum() if len(seg_len) else 1
    acc = np.concatenate([[0], np.cumsum(seg_len)])
    w_end = w if w_end is None else w_end
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        r0 = (w + (w_end - w) * acc[i] / total) / 2
        r1 = (w + (w_end - w) * acc[i + 1] / total) / 2
        R = max(r0, r1) + 2
        xa, xb = int(min(p0[0], p1[0]) - R + X0), int(max(p0[0], p1[0]) + R + X0) + 1
        ya, yb = int(YT - max(p0[1], p1[1]) - R), int(YT - min(p0[1], p1[1]) + R) + 1
        xa, ya = max(xa, 0), max(ya, 0)
        xb, yb = min(xb, W), min(yb, H)
        px, py = FX[ya:yb, xa:xb], FY[ya:yb, xa:xb]
        d = p1 - p0
        L2 = float(d @ d) or 1e-6
        t = np.clip(((px - p0[0]) * d[0] + (py - p0[1]) * d[1]) / L2, 0, 1)
        qx, qy = p0[0] + t * d[0], p0[1] + t * d[1]
        dist = np.hypot(px - qx, py - qy)
        out[ya:yb, xa:xb] |= dist <= (r0 + (r1 - r0) * t)
    return out

def stroke(pts, w=170, w_end=None):
    return capsule_chain(dense(pts), w, w_end)

def dense(pts, step=6):
    """Densify polyline (keeps corners)."""
    pts = np.asarray(pts, np.float32)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)

def arc(cx, cy, rx, ry, a0, a1, n=None):
    """Points along an elliptical arc, angles in degrees (CCW, 0 = +x)."""
    n = n or max(8, int(abs(a1 - a0) / 4))
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)

def ellipse(cx, cy, rx, ry=None, angle=0):
    ry = rx if ry is None else ry
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    dx, dy = FX - cx, FY - cy
    u, v = dx * c + dy * s, -dx * s + dy * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1

def rect(x0, y0, x1, y1):
    return (FX >= x0) & (FX <= x1) & (FY >= y0) & (FY <= y1)

def poly(pts):
    m = np.zeros((H, W), np.uint8)
    p = np.array([[x + X0, YT - y] for x, y in pts], np.int32)
    cv2.fillPoly(m, [p], 1)
    return m.astype(bool)

def halfplane(a, b, c):
    """a*x + b*y + c >= 0"""
    return a * FX + b * FY + c >= 0

# ---------- placing raster masks (originals) ----------
def place(mask, x=0, y=0, sx=1.0, sy=1.0, flipx=False, flipy=False, rot=0):
    """Put a bbox-cropped mask (bottom-left at (x,y)) onto canvas, optional scale/flip/rotate about its center."""
    m = mask.astype(np.float32)
    if flipx: m = m[:, ::-1]
    if flipy: m = m[::-1, :]
    if sx != 1 or sy != 1:
        m = cv2.resize(m, (max(1, round(m.shape[1] * sx)), max(1, round(m.shape[0] * sy))), interpolation=cv2.INTER_LINEAR)
    if rot:
        h, w = m.shape
        pad = int(max(h, w) * 0.3)
        m = np.pad(m, pad)
        M = cv2.getRotationMatrix2D((m.shape[1] / 2, m.shape[0] / 2), rot, 1)
        m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(m > 0.5)
        m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    out = np.zeros((H, W), np.float32)
    h, w = m.shape
    r0, c0 = YT - y - h, X0 + x
    out[r0:r0 + h, c0:c0 + w] = m
    return out > 0.5

def bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min() - X0, YT - ys.max(), xs.max() - X0, YT - ys.min()   # x0,y0,x1,y1 font coords

def crop(m):
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

# ---------- finishing ----------
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def round_corners(m, r=30):
    """Opening: rounds convex corners."""
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(r)).astype(bool)

def soften_inner(m, r=12):
    """Closing: rounds concave corners (careful: fills gaps < 2r)."""
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(r)).astype(bool)

def wobble(m, seed, amp=7.0, scale=55.0, rot_deg=0.0):
    """Hand-cut irregularity: smooth random displacement field + optional tiny rotation."""
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(ys.min() - 60, 0), min(ys.max() + 60, H), max(xs.min() - 60, 0), min(xs.max() + 60, W)
    sub = m[y0:y1, x0:x1].astype(np.float32)
    h, w = sub.shape
    def field():
        f = ndi.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), scale)
        return f / (f.std() + 1e-9) * amp
    dx, dy = field(), field()
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    if rot_deg:
        cy, cx = h / 2, w / 2
        a = np.radians(rot_deg)
        rx, ry = gx - cx, gy - cy
        gx2 = cx + rx * np.cos(a) - ry * np.sin(a)
        gy2 = cy + rx * np.sin(a) + ry * np.cos(a)
        gx, gy = gx2, gy2
    warped = cv2.remap(sub, (gx + dx).astype(np.float32), (gy + dy).astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = warped > 0.5
    return out

def blur_thresh(m, sigma=4):
    return ndi.gaussian_filter(m.astype(np.float32), sigma) > 0.5

def finish(m, name, clip=(0, CAP), corner=34, amp=4.5, scale=150.0, rot=None, sigma=6, inner=None, fair=12):
    """Standard finishing for constructed glyphs: clip to baseline/cap, round corners, wobble, smooth."""
    if clip is not None:
        m = m & (FY >= clip[0]) & (FY <= clip[1])
    if corner:
        m = round_corners(m, int(corner * 1.3))
    if inner is None:
        inner = 16 if corner else 0   # punctuation/accents (corner=0) keep their open shapes
    if inner:
        m = soften_inner(m, inner)
    s = seed_of(name)
    if rot is None:
        rot = (np.random.default_rng(s + 1).random() - 0.5) * 1.6
    if amp:
        m = wobble(m, s, amp, scale, rot)
    m = blur_thresh(m, sigma)
    if fair:
        m = polish(m, fair, hole_round=None)
    return m

# ---------- polishing ----------
def _resample_closed(p, step):
    q = np.vstack([p, p[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    n = max(16, int(round(s[-1] / step)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)

def fair_closed(p, sigma, step):
    """Low-pass a closed polyline with the (2G - G*G) filter: removes bumps shorter than ~sigma
    while barely shrinking the overall shape (no 'melting' of slits and corners)."""
    s = sigma / step
    g = lambda a: ndi.gaussian_filter1d(a, s, axis=0, mode='wrap')
    a = g(p)
    return 2 * a - g(a)

def polish(m, sigma=18, step=2.0, hole_round=0.6, min_area=80):
    """Redraw a mask with faired outer contours and clean elliptical counters."""
    cnts, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = np.zeros((H, W), np.uint8)
    if hier is None: return out.astype(bool)
    holes = []
    for i, c in enumerate(cnts):
        p = c[:, 0, :].astype(np.float64) + 0.5
        if cv2.contourArea(c) < min_area: continue
        if hier[0][i][3] >= 0:
            holes.append((c, p)); continue
        q = fair_closed(_resample_closed(p, step), sigma, step)
        cv2.fillPoly(out, [np.round(q * 16).astype(np.int32)], 1, shift=4)
    for c, p in holes:
        if hole_round is None:
            q = fair_closed(_resample_closed(p, step), sigma * 0.5, step)
            cv2.fillPoly(out, [np.round(q * 16).astype(np.int32)], 0, shift=4); continue
        mo = cv2.moments(c)
        A = abs(cv2.contourArea(c)) + len(c) * 0.5
        cx, cy = mo['m10'] / mo['m00'], mo['m01'] / mo['m00']
        cov = np.array([[mo['mu20'], mo['mu11']], [mo['mu11'], mo['mu02']]]) / mo['m00']
        ev, evec = np.linalg.eigh(cov)
        a, b = 2 * np.sqrt(max(ev[1], 1)), 2 * np.sqrt(max(ev[0], 1))
        gm = np.sqrt(a * b)
        a, b = a + (gm - a) * hole_round, b + (gm - b) * hole_round
        k = np.sqrt(A / (np.pi * a * b)); a, b = a * k, b * k
        ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
        cv2.ellipse(out, (int(round(cx * 16)), int(round(cy * 16))), (int(round(a * 16)), int(round(b * 16))),
                    ang, 0, 360, 0, -1, shift=4)
    return out.astype(bool)
