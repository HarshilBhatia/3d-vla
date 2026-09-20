import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
methods=[{"id":"base_s160k","checkpoint":"train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth"},{"id":"deltam_s140k","checkpoint":"train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth"},{"id":"video_pooled_s084k","checkpoint":"train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth","overrides":{"visual_num_history":5,"eval_proprio_history_order":"past_to_current"}},{"id":"video_fullpatch_best","checkpoint":"train_logs/PerAct2/peract2_orbital_vid_deltam_fullpatch_finetune_300k_a6000_b64/best.pth","overrides":{"visual_num_history":5,"eval_proprio_history_order":"past_to_current"}}]
for src,setting in [('idlt_task_seen_group_v01.json','task_seen_group'),('idlt_task_heldout_group_v01.json','task_heldout_group')]:
 p=json.loads((ROOT/'instructions/eval_plans'/src).read_text());p['campaign_id']=f'idlt_scaled_extreme_{setting}_v02_5060';p['calibration_registry']='instructions/eval_calibrations_idlt_scaled_extreme_v01.json';p['methods']=methods;p['calibrations']=['unknown-e50deg-t50cm','unknown-e60deg-t60cm'];p['task_calibrations']={}
 for t in p['tasks']:
  g=p['task_viewpoints'][t]['spawn_camera_group'].lower();p['task_calibrations'][t]=[f'seen-{g}-medium-unknown-e50deg-t50cm-v01',f'seen-{g}-medium-unknown-e60deg-t60cm-v01']
 (ROOT/'instructions/eval_plans'/f'idlt_scaled_extreme_{setting}_v01.json').write_text(json.dumps(p,indent=2)+'\n')
