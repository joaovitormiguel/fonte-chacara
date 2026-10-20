import cv2, numpy as np
im=cv2.imread('work/placa_rot.png')
lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32)
L=lab[...,0]; b=lab[...,2]
score=(255-L)+0.8*(200-b)
score=cv2.GaussianBlur(score,(0,0),0.8)
sc=np.clip((score-score.min())/(score.max()-score.min())*255,0,255).astype(np.uint8)
T=109
m=(sc>T).astype(np.uint8)*255
m[:, :85]=0; m[:45,:]=0; m[715:,:]=0
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
mo=cv2.morphologyEx(m,cv2.MORPH_OPEN,k)
n,lbl,st,cen=cv2.connectedComponentsWithStats(mo)
vis=im.copy()
for i in range(1,n):
    x,y,w,h,a=st[i]
    if a<300: continue
    print(i,x,y,w,h,a)
    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.putText(vis,str(i),(x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1)
# small pieces (accents) with smaller opening
k2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
mo2=cv2.morphologyEx(m,cv2.MORPH_OPEN,k2)
for (x0,y0,x1,y1,name) in [(460,120,520,170,'acute'),(560,300,620,335,'tilde')]:
    sub=mo2[y0:y1,x0:x1]
    nn,l2,s2,c2=cv2.connectedComponentsWithStats(sub)
    for j in range(1,nn):
        print(name,j,s2[j][0]+x0,s2[j][1]+y0,s2[j][2],s2[j][3],s2[j][4])
cv2.imwrite('work/cc2.png',vis)
np.save('work/score.npy',sc)
