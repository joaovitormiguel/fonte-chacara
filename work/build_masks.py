import sys, os, json, cv2, numpy as np, time
sys.path.insert(0, 'work')
from lib import *
import registry
t=time.time()
os.makedirs('work/masks', exist_ok=True)
ms = registry.build_all()
meta = {}
for nm, m in ms.items():
    x0, y0, x1, y1 = bbox(m)
    cv2.imwrite(f'work/masks/{nm}.png', (crop(m) * 255).astype(np.uint8))
    meta[nm] = [int(x0), int(y0), int(x1), int(y1)]
json.dump(meta, open('work/masks/meta.json', 'w'), indent=0)
print(len(ms), 'glyphs in', round(time.time()-t,1), 's')
