# grogu status

Updated 2026-10-14 20:46 EDT at f2d397b.

| job | run | state | node | GPU | progress (this launch) | speed |
|---|---|---|---|---|---|---|
| 4030292 | exp10b_chunk8_fullpatch_refine | PENDING | - | - |  | - |
| 4030291 | exp10b_chunk8_fullpatch_refine | RUNNING | grogu-2-30 | A5000 | 19969/38200 | 1.08it/s |
| 4030294 | exp10c_chunk8_fullpatch_refine_x0 | PENDING | - | - |  | - |
| 4030293 | exp10c_chunk8_fullpatch_refine_x0 | RUNNING | grogu-2-30 | A5000 | 20748/38200 | 1.18it/s |
| 4030177 | exp10d_chunk8_fullpatch_refine_rel | PENDING | - | - |  | - |
| 4030176 | exp10d_chunk8_fullpatch_refine_rel | RUNNING | grogu-2-25 | A5000 | 13338/73900 | 1.15it/s |
| 4030179 | exp10f_chunk8_pooled_refine | PENDING | - | - |  | - |
| 4030178 | exp10f_chunk8_pooled_refine | RUNNING | grogu-2-25 | A5000 | 16323/75500 | 1.37it/s |

GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded.

| run | newest interm_step | last.pth iter |
|---|---|---|
| peract2_orbital_handover_dense_chunk8_deltam | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine | interm_step_42500.pth | 44200 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_rel | interm_step_35000.pth | 39400 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_x0 | interm_step_45000.pth | 45000 |
| peract2_orbital_handover_dense_chunk8_pooled_refine | interm_step_40000.pth | 40800 |
| peract2_orbital_handover_dense_chunk8_rel | interm_step_50000.pth | 50000 |
