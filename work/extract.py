import cv2, numpy as np, os
from scipy import ndimage as ndi
sc=np.load('work/score.npy').astype(np.float32)
T=109
m=(sc>T).astype(np.uint8)*255
m[:, :85]=0; m[:45,:]=0; m[715:,:]=0
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
mo=cv2.morphologyEx(m,cv2.MORPH_OPEN,k)
n,lbl,st,cen=cv2.connectedComponentsWithStats(mo)
k2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
mo2=cv2.morphologyEx(m,cv2.MORPH_OPEN,k2)

names={6:'C1',7:'H',5:'A1',10:'C2',9:'A2',8:'R1',11:'A3',16:'S1',18:'A4',17:'O1',
       19:'F',22:'R2',26:'A5',23:'N',20:'C3',25:'I',24:'S2',21:'C4',27:'O2'}
S=8
os.makedirs('work/raw',exist_ok=True)
def save(name, compmask, pad=6):
    ys,xs=np.nonzero(compmask)
    x0,x1,y0,y1=xs.min()-pad,xs.max()+pad+1,ys.min()-pad,ys.max()+pad+1
    region=cv2.dilate(compmask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))[y0:y1,x0:x1]
    crop=sc[y0:y1,x0:x1]
    up=cv2.resize(crop,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC)
    up=cv2.GaussianBlur(up,(0,0),S*0.6)
    reg=cv2.resize(region,None,fx=S,fy=S,interpolation=cv2.INTER_LINEAR)>0.5
    bw=(up>T)&reg
    # keep largest component
    l,nn=ndi.label(bw)
    if nn>1:
        sizes=ndi.sum(bw,l,range(1,nn+1)); bw=l==(np.argmax(sizes)+1)
    # fill small holes (wood texture), keep counters
    filled=ndi.binary_fill_holes(bw)
    holes=filled&~bw
    hl,hn=ndi.label(holes)
    keep=np.zeros_like(bw)
    for j in range(1,hn+1):
        a=(hl==j).sum()
        if a> 2500: keep|=hl==j
    bw=filled&~keep
    cv2.imwrite(f'work/raw/{name}.png',(bw*255).astype(np.uint8))
    print(name,(x0,y0,x1,y1),bw.shape, 'holes kept', int(keep.any()))
for i,nm in names.items():
    c=(lbl==i)
    if nm=='F': c[:, :130]=False
    save(nm,c)
# accents
for (x0,y0,x1,y1,name) in [(460,128,520,159,'acute'),(555,312,620,336,'tilde')]:
    sub=np.zeros_like(mo2); sub[y0:y1,x0:x1]=mo2[y0:y1,x0:x1]
    l,nn=ndi.label(sub>0); sizes=ndi.sum(sub>0,l,range(1,nn+1))
    save(name,l==(np.argmax(sizes)+1),pad=4)
