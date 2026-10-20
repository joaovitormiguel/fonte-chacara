import cv2, numpy as np, os, json
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
os.makedirs('work/norm',exist_ok=True)
CAP=700
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def smooth(bw, sigma, oc=0):
    a=bw.astype(np.uint8)*255
    if oc:
        a=cv2.morphologyEx(a,cv2.MORPH_OPEN,disk(oc)); a=cv2.morphologyEx(a,cv2.MORPH_CLOSE,disk(oc))
    a=cv2.GaussianBlur(a.astype(np.float32),(0,0),sigma)
    return a>127
# per-glyph cleaning strength (in x8 raw pixels)
cfg={'A1':(10,12),'A2':(8,8),'A3':(10,16),'A4':(8,8),'A5':(10,12),
     'C1':(14,24),'C2':(8,8),'C3':(9,12),'C4':(9,12),
     'F':(12,22),'H':(12,22),'I':(8,8),'N':(9,12),'O1':(8,8),
     'R1':(9,10),'R2':(10,18),'S1':(8,8),'S2':(8,8),'acute':(8,8),'tilde':(8,8)}
meta={}
for name,(sig,oc) in cfg.items():
    bw=cv2.imread(f'work/raw/{name}.png',0)>127
    bw=np.pad(bw,60)
    bw=smooth(bw,sig,oc)
    ys,xs=np.nonzero(bw)
    h=ys.max()-ys.min()+1
    sc=CAP/h if name not in ('acute','tilde') else 0.7*CAP/124*1.0  # accents keep sign proportions (ref A height ~124px*8)
    if name in ('acute','tilde'): sc=CAP/(124*8)
    out=cv2.resize(bw.astype(np.float32),None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
    out=cv2.GaussianBlur(out,(0,0),1.2)>0.5
    ys,xs=np.nonzero(out)
    out=out[ys.min():ys.max()+1, xs.min():xs.max()+1]
    cv2.imwrite(f'work/norm/{name}.png',(out*255).astype(np.uint8))
    # stroke width estimate: 2*DT at skeleton
    dt=ndi.distance_transform_edt(out)
    sk=skeletonize(out)
    w=2*dt[sk]
    meta[name]=dict(w=int(out.shape[1]),h=int(out.shape[0]),stroke_med=round(float(np.median(w)),1),stroke_p75=round(float(np.percentile(w,75)),1))
    print(name,meta[name])
json.dump(meta,open('work/norm/meta.json','w'),indent=1)
