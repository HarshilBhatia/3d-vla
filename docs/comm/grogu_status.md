# grogu status

Updated 2026-10-14 14:57 EDT at 66b0ee2.

| job | run | state | node | GPU | progress (this launch) | speed |
|---|---|---|---|---|---|---|
| 4030169 | exp10a_chunk8_deltam | PENDING | - | - |  | - |
| 4030168 | exp10a_chunk8_deltam | RUNNING | grogu-2-30 | A5000 | 28154/30100 | 3.07it/s |
| 4030173 | exp10b_chunk8_fullpatch_refine | PENDING | - | - |  | - |
| 4030172 | exp10b_chunk8_fullpatch_refine | RUNNING | grogu-2-30 | A5000 | 6260/83700 | 1.20it/s |
| 4030175 | exp10c_chunk8_fullpatch_refine_x0 | PENDING | - | - |  | - |
| 4030174 | exp10c_chunk8_fullpatch_refine_x0 | RUNNING | grogu-2-30 | A5000 | 6265/83700 | 1.17it/s |
| 4030177 | exp10d_chunk8_fullpatch_refine_rel | PENDING | - | - |  | - |
| 4030176 | exp10d_chunk8_fullpatch_refine_rel | RUNNING | grogu-4-8 | rtx3090 | 5699/83700 | 1.12it/s |
| 4030179 | exp10f_chunk8_pooled_refine | PENDING | - | - |  | - |
| 4030178 | exp10f_chunk8_pooled_refine | RUNNING | grogu-4-8 | rtx3090 | 3089/83300 | 1.42it/s |

GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded.

| run | newest interm_step | last.pth iter |
|---|---|---|
| peract2_orbital_handover_dense_chunk8_deltam | interm_step_40000.pth | 48100 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine | interm_step_20000.pth | 22500 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_rel | interm_step_20000.pth | 22000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_x0 | interm_step_20000.pth | 22500 |
| peract2_orbital_handover_dense_chunk8_pooled_refine | interm_step_15000.pth | 19800 |
| peract2_orbital_handover_dense_chunk8_rel | interm_step_50000.pth | 50000 |
