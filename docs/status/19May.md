### Experiments I need to run

baselines:
1. my VLA with 2D? - NoPE, use standard vlas lol. 


1. constant (camera) miscalibration + add random noise -- ALL Labs are now miscalibrated. Random noise is just a pertubation technique for us now (interesting).
    a) 3DFA (baseline) 
    b) DeltaM + EE(ours)
    This is run on the subset first.
    Start zarr dataset transfer! 
RUNNING on DeltaAI + Delta. 


better baseline? -- 2D diffuse actor?


TODO: implement noisy calibration for online eval. = DONE.

1.1 valuate this on 
a) clean calibration 
b) noisy calibration (but fixed.)


2. Process dataset in the following manner (full).
100 episodes = train, 20 episodes val.
so currently out of 120, only use 100 for train (33,33,34.).
Have different miscalibration for val-labs! 

3. train the random miscal lab setup.


4. Peract2? PSC for data collection or what ..? 


### Other things to do by tonight? 
1. writeup everything on overleaf.
2. finalise experiments. 
3. prepare the dataset (full finetune)



Online eval (simulated env).

To strengthen it further: can you evaluate B and D on out-of-distribution noise levels (e.g. train on ≤5°, test on 10°)? If D degrades more gracefully than B, that's strong evidence the head is learning genuine geometric adaptation, not just noise tolerance.



Open Questions:
1. Is this specific to flow policy? - Yes / No idk. 
2. Are there other ways to have adaptive conditioning instead of deltaM + 3D RoPE?
3. Should we use 6x6 matrix or the full DxD matrix? (size ablation?)
. 


How to do so many experiments? 
1. run everything on 2 nodes with aggressive checkpointing!!

TODO
1. clean up code and get rid of things that don't work.
