import json, copy
from pathlib import Path
import numpy as np
from evaluation.planning.materialize_seen_base_residual_registry import axis_angle_to_rotation, transform
ROOT=Path(__file__).resolve().parents[2]
base=json.loads((ROOT/'instructions/orbital_miscalibration_noise_bimanual_external_only.json').read_text())
noise=json.loads((ROOT/'instructions/random_miscal_noise_bimanual.json').read_text())
cameras=base['cameras']; groups=base['groups']; out={"schema_version":1,"camera_order":cameras,"composition":"T_observed = (epsilon @ delta_group) @ T_true","source":{"base_file":"instructions/orbital_miscalibration_noise_bimanual_external_only.json","base_level":"medium","residual_file":"instructions/random_miscal_noise_bimanual.json","residual_camera_ids":[0,1],"scaling":"30/30 and 40/40 use the fixed 20/20 residual directions scaled by 1.5x and 2x"},"realizations":{}}
out['realizations']['calibrated']={"description":"Identity calibration realization.","transforms":{c:np.eye(4).tolist() for c in cameras}}
for g in groups:
  for label, factor in [('e30deg-t30cm',1.5),('e40deg-t40cm',2.0),('e50deg-t50cm',2.5),('e60deg-t60cm',3.0)]:
    ts={}
    for i,c in enumerate(cameras):
      b=transform(base['levels']['medium'][g].get(c)); e=np.eye(4)
      if i<2:
        aa=np.asarray(noise['rotation']['20deg'][c]['axis_angle_rad'])*factor
        tr=np.asarray(noise['translation']['20cm'][c]['translation_m'])*factor
        e[:3,:3]=axis_angle_to_rotation(aa); e[:3,3]=tr
      ts[c]=(e@b).tolist()
    rid=f"seen-{g.lower()}-medium-unknown-{label}-v01"
    out['realizations'][rid]={"description":f"Fixed medium base for {g} plus scaled 20/20 residual ({label}).","transforms":ts}
(ROOT/'instructions/eval_calibrations_idlt_scaled_extreme_v01.json').write_text(json.dumps(out,indent=2)+'\n')
print('wrote',len(out['realizations']))
