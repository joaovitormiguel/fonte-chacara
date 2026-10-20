"""Mask -> smooth cubic Bezier contours (Schneider curve fitting)."""
import cv2, numpy as np
from scipy import ndimage as ndi
from lib import X0, YT

def resample_closed(p, step):
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(12, int(round(L / step)))
    t = np.linspace(0, L, n, endpoint=False)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)

def smooth_closed(p, sigma_pts):
    if sigma_pts <= 0: return p
    g = lambda a: ndi.gaussian_filter1d(a, sigma_pts, axis=0, mode='wrap')
    a = g(p)
    return 2 * a - g(a)

# ---- Schneider ----
def bez(ctrl, t):
    t = t[:, None]; mt = 1 - t
    return mt**3*ctrl[0] + 3*mt**2*t*ctrl[1] + 3*mt*t**2*ctrl[2] + t**3*ctrl[3]
def bez_d1(c, t):
    t = t[:, None]; mt = 1 - t
    return 3*mt**2*(c[1]-c[0]) + 6*mt*t*(c[2]-c[1]) + 3*t**2*(c[3]-c[2])
def bez_d2(c, t):
    t = t[:, None]
    return 6*(1-t)*(c[2]-2*c[1]+c[0]) + 6*t*(c[3]-2*c[2]+c[1])

def chord_param(d):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(d, axis=0), axis=1))])
    return s / s[-1] if s[-1] > 0 else np.linspace(0, 1, len(d))

def gen_bezier(d, u, t1, t2):
    p0, p3 = d[0], d[-1]
    A1 = (3*(1-u)**2*u)[:, None] * t1
    A2 = (3*(1-u)*u**2)[:, None] * t2
    C = np.array([[np.sum(A1*A1), np.sum(A1*A2)], [np.sum(A1*A2), np.sum(A2*A2)]])
    base = bez(np.array([p0, p0, p3, p3]), u)
    tmp = d - base
    X = np.array([np.sum(A1*tmp), np.sum(A2*tmp)])
    det = C[0,0]*C[1,1] - C[0,1]*C[1,0]
    seg = np.linalg.norm(p3 - p0)
    if abs(det) > 1e-12:
        a1 = (X[0]*C[1,1] - X[1]*C[0,1]) / det
        a2 = (C[0,0]*X[1] - C[1,0]*X[0]) / det
    else:
        a1 = a2 = seg / 3
    eps = 1e-6 * seg
    if a1 < eps or a2 < eps or a1 > seg*1.5 or a2 > seg*1.5:
        a1 = a2 = seg / 3
    return np.array([p0, p0 + t1*a1, p3 + t2*a2, p3])

def reparam(c, d, u):
    q = bez(c, u) - d
    d1, d2 = bez_d1(c, u), bez_d2(c, u)
    num = np.sum(q*d1, 1); den = np.sum(d1*d1, 1) + np.sum(q*d2, 1)
    nu = u - np.where(np.abs(den) > 1e-12, num/den, 0)
    return np.clip(nu, 0, 1)

def max_err(c, d, u):
    e = np.sum((bez(c, u) - d)**2, 1)
    i = int(np.argmax(e))
    return e[i], i

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-12 else v

def fit_cubic(d, t1, t2, err, out, depth=0):
    if len(d) == 2:
        dist = np.linalg.norm(d[1]-d[0]) / 3
        out.append(np.array([d[0], d[0]+t1*dist, d[1]+t2*dist, d[1]])); return
    u = chord_param(d)
    c = gen_bezier(d, u, t1, t2)
    e, i = max_err(c, d, u)
    if e < err*err:
        out.append(c); return
    if e < (err*4)**2:
        for _ in range(6):
            u = reparam(c, d, u)
            c = gen_bezier(d, u, t1, t2)
            e, i = max_err(c, d, u)
            if e < err*err:
                out.append(c); return
    i = min(max(i, 1), len(d)-2)
    tc = unit(d[i-1] - d[i+1])
    fit_cubic(d[:i+1], t1, tc, err, out, depth+1)
    fit_cubic(d[i:], -tc, t2, err, out, depth+1)

def fit_closed(p, err=1.5, nchunks=6):
    """Fit closed polyline with G1-continuous cubic segments."""
    n = len(p)
    # split at points of highest curvature (well spaced) for good joins
    k = max(3, min(nchunks, n // 6))
    v1 = np.roll(p, -1, 0) - p; v0 = p - np.roll(p, 1, 0)
    ang = np.abs(np.arctan2(v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0], np.sum(v0*v1, 1)))
    ang = ndi.gaussian_filter1d(ang, 2, mode='wrap')
    idx = []
    order = np.argsort(-ang)
    mind = n // (k * 2)
    for j in order:
        if all(min(abs(j-q), n-abs(j-q)) >= mind for q in idx):
            idx.append(int(j))
        if len(idx) >= k: break
    idx.sort()
    tang = lambda j: unit(p[(j+1) % n] - p[(j-1) % n])
    segs = []
    for a, b in zip(idx, idx[1:] + [idx[0] + n]):
        d = np.array([p[j % n] for j in range(a, b + 1)])
        fit_cubic(d, tang(a % n), -tang(b % n), err, segs)
    return segs

def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y)

def mask_to_beziers(m, smooth_units=3.0, step=2.5, err=1.6, min_area=60):
    """Returns list of contours, each a list of cubic segments (4x2 arrays) in font units.
    Outer contours CCW, holes CW (PostScript convention)."""
    m8 = m.astype(np.uint8)
    cnts, hier = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    res = []
    if hier is None: return res
    for ci, c in enumerate(cnts):
        pts = c[:, 0, :].astype(np.float64)
        if len(pts) < 8: continue
        fp = np.stack([pts[:, 0] - X0 + 0.5, YT - pts[:, 1] - 0.5], 1)
        is_hole = hier[0][ci][3] >= 0
        if abs(signed_area(fp)) < min_area: continue
        q = resample_closed(fp, step)
        q = smooth_closed(q, smooth_units / step)
        a = signed_area(q)
        want_ccw = not is_hole
        if (a > 0) != want_ccw:
            q = q[::-1]
        segs = fit_closed(q, err)
        res.append(segs)
    return res

def render_beziers(contours, scale=0.25, pad=40):
    """Rasterize contours for preview (returns bool mask in font units*scale, y-up handled)."""
    allp = np.vstack([np.vstack(s) for s in contours])
    x0, y0 = allp.min(0) - pad; x1, y1 = allp.max(0) + pad
    Wd, Hd = int((x1-x0)*scale)+1, int((y1-y0)*scale)+1
    img = np.zeros((Hd, Wd), np.uint8)
    polys = []
    t = np.linspace(0, 1, 16)
    for segs in contours:
        pts = np.vstack([bez(s, t) for s in segs])
        polys.append(np.round(np.stack([(pts[:,0]-x0)*scale, (y1-pts[:,1])*scale], 1)*16).astype(np.int32))
    cv2.fillPoly(img, polys, 255, lineType=cv2.LINE_AA, shift=4)
    return img
