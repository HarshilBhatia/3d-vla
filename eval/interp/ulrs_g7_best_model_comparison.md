# ULRS G7 — best model result per calibration condition

| Model | Clean | External 2° / 2 cm | External 5° / 5 cm | External 10° / 10 cm |
|---|---:|---:|---:|---:|
| Base | 72.4% (base_s160k) | 71.5% (base_s160k) | 71.3% (base_s160k) | 71.9% (base_s160k) |
| DeltaM | 75.5% (deltam_s140k) | 75.4% (deltam_s140k) | 75.1% (deltam_s140k) | 74.7% (deltam_s140k) |
| Video-DeltaM | 78.5% (video_deltam_best_s084k) | 78.3% (video_deltam_best_s084k) | 78.8% (video_deltam_s100k) | 78.5% (video_deltam_s100k) |
| Base PerAct2 reference | 85.0% | 85.0% | 85.0% | 85.0% |
| Cam-var PerAct2 reference | 79.2% | 79.2% | 79.2% | 79.2% |

Base/DeltaM points are re-aggregated from the ULRS raw task JSONs; Video-DeltaM points are re-aggregated from the decision-grade ULRS rows in `docs/results/video_deltam_task_metrics.csv`. The two PerAct2 references were supplied as horizontal reference levels and are not asserted to share this G7 external-camera protocol.
