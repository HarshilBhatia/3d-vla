# Experiment Log

| Job ID   | Cluster | Job Name                 | Experiment Config                      | Notes                                              | Miscal                                      | Status             |
|----------|---------|--------------------------|----------------------------------------|----------------------------------------------------|---------------------------------------------|--------------------|
| 18343783 | delta   | multicam_default         | —                                      | Baseline 3DFA, no extrinsics prediction            | fixed medium per-group + randnoise (≤3°, ≤1cm) | RUNNING            |
| 18343790 | delta   | multicam_deltaM_med      | orb_deltaM_full_fixed_medium_randnoise | DeltaM full (DxD), fixed medium miscal + randnoise | fixed medium per-group + randnoise (≤3°, ≤1cm) | RUNNING            |
| 18346455 | delta   | multicam_deltaM_6x6_med  | orb_deltaM_fixed_medium_randnoise      | DeltaM 6x6, fixed medium miscal + randnoise        | fixed medium per-group + randnoise (≤3°, ≤1cm) | RUNNING ON DELTAAI |
