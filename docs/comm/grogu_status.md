# grogu status

Updated 2026-10-14 19:44 EDT at 10c2cdc.

| job | run | state | node | GPU | progress (this launch) | speed |
|---|---|---|---|---|---|---|
| 4030292 | exp10b_chunk8_fullpatch_refine | PENDING | - | - |  | - |
| 4030291 | exp10b_chunk8_fullpatch_refine | RUNNING | grogu-2-30 | A5000 | 16051/38200 | 1.10it/s |
| 4030294 | exp10c_chunk8_fullpatch_refine_x0 | PENDING | - | - |  | - |
| 4030293 | exp10c_chunk8_fullpatch_refine_x0 | RUNNING | grogu-2-30 | A5000 | 16658/38200 | 1.16it/s |
| 4030177 | exp10d_chunk8_fullpatch_refine_rel | PENDING | - | - |  | - |
| 4030176 | exp10d_chunk8_fullpatch_refine_rel | RUNNING | grogu-2-25 | A5000 | 9337/73900 | 1.19it/s |
| 4030179 | exp10f_chunk8_pooled_refine | PENDING | - | - |  | - |
| 4030178 | exp10f_chunk8_pooled_refine | RUNNING | grogu-2-25 | A5000 | 11696/75500 | 1.51it/s |

GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded.

| run | newest interm_step | last.pth iter |
|---|---|---|
| peract2_orbital_handover_dense_chunk8_deltam | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine | interm_step_40000.pth | 40300 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_rel | interm_step_35000.pth | 35400 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_x0 | interm_step_40000.pth | 40900 |
| peract2_orbital_handover_dense_chunk8_pooled_refine | interm_step_35000.pth | 36200 |
| peract2_orbital_handover_dense_chunk8_rel | interm_step_50000.pth | 50000 |
