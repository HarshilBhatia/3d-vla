import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
cams=json.loads((ROOT/'instructions/orbital_cameras_grouped.json').read_text()); g=next(x for x in cams if x['group']=='G7')
p0=np.asarray(g['left']['pos'],float); p1=np.asarray(g['right']['pos'],float); d=p0-p1; d=d/np.linalg.norm(d)
I=np.eye(4); realizations={"g7-clean-v01":{"description":"Identity calibration.","transforms":{c:I.tolist() for c in ['orbital_left','orbital_right','wrist_left','wrist_right']}}}
for cm in [10,20,40,60]:
  ts={c:I.tolist() for c in ['orbital_left','orbital_right','wrist_left','wrist_right']}
  a=I.copy(); b=I.copy(); a[:3,3]=(d*cm/100).tolist(); b[:3,3]=(-d*cm/100).tolist(); ts['orbital_left']=a.tolist();ts['orbital_right']=b.tolist()
  realizations[f'g7-opposite-{cm}cm-v01']={"description":f"G7 external cameras translated {cm}cm away from each other; no rotation.","transforms":ts}
out={"schema_version":1,"camera_order":["orbital_left","orbital_right","wrist_left","wrist_right"],"composition":"T_observed = T_true with direct opposing external-camera translations","source":{"camera_group":"G7","direction":"from camera 1 to camera 0; camera 0 moves +d and camera 1 moves -d","rotation":"identity"},"realizations":realizations}
(ROOT/'instructions/eval_calibrations_g7_opposite_camera_v01.json').write_text(json.dumps(out,indent=2)+'\n')
print('outward direction',d.tolist())
