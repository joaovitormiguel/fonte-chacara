import sys, json, cv2, numpy as np
sys.path.insert(0,'work')
meta=json.load(open('work/masks/meta.json'))
def get(nm):
    return cv2.imread(f'work/masks/{nm}.png',0)>127, meta[nm]
def sheet(names, out, per_row=10, s=0.22, label=True):
    cell_h=int(1150*s); tiles=[]
    for nm in names:
        m,(x0,y0,x1,y1)=get(nm)
        a=np.where(m,0,255).astype(np.uint8)
        a=cv2.resize(a,(max(1,int(a.shape[1]*s)),max(1,int(a.shape[0]*s))),interpolation=cv2.INTER_AREA)
        w=a.shape[1]+24
        t=np.full((cell_h+26,max(w,60)),255,np.uint8)
        base=26+int(900*s)
        top=base-int(y1*s)
        t[top:top+a.shape[0],12:12+a.shape[1]]=a[:t.shape[0]-top]
        cv2.line(t,(0,base),(t.shape[1],base),190,1); cv2.line(t,(0,base-int(700*s)),(t.shape[1],base-int(700*s)),225,1)
        if label: cv2.putText(t,nm,(3,16),cv2.FONT_HERSHEY_SIMPLEX,0.42,0,1)
        tiles.append(t)
    rows=[np.hstack(tiles[i:i+per_row]) for i in range(0,len(tiles),per_row)]
    W=max(r.shape[1] for r in rows)
    cv2.imwrite(out,np.vstack([cv2.copyMakeBorder(r,0,0,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=255) for r in rows]))
if __name__=='__main__':
    names=[n for n in meta if '.alt' not in n]
    sheet(names,'work/preview_all.png',per_row=13)
