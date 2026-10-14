# grogu status

Updated 2026-10-14 16:30 EDT at eadb50d.

| job | run | state | node | GPU | progress (this launch) | speed |
|---|---|---|---|---|---|---|
| 4030292 | exp10b_chunk8_fullpatch_refine | PENDING | - | - |  | - |
| 4030291 | exp10b_chunk8_fullpatch_refine | RUNNING | grogu-2-30 | A5000 | 3699/38200 | 1.13it/s |
| 4030294 | exp10c_chunk8_fullpatch_refine_x0 | PENDING | - | - |  | - |
| 4030293 | exp10c_chunk8_fullpatch_refine_x0 | RUNNING | grogu-2-30 | A5000 | 3736/38200 | 1.16it/s |
| 4030176 | exp10d_chunk8_fullpatch_refine_rel | PENDING | - | - | 9832/83700 | 1.14it/s |
| 4030177 | exp10d_chunk8_fullpatch_refine_rel | PENDING | - | - |  | - |
| 4030178 | exp10f_chunk8_pooled_refine | PENDING | - | - | 7820/83300 | 1.39it/s |
| 4030179 | exp10f_chunk8_pooled_refine | PENDING | - | - |  | - |

GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded.

| run | newest interm_step | last.pth iter |
|---|---|---|
| peract2_orbital_handover_dense_chunk8_deltam | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine | interm_step_27500.pth | 28000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_rel | interm_step_25000.pth | 26100 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_x0 | interm_step_27500.pth | 28000 |
| peract2_orbital_handover_dense_chunk8_pooled_refine | interm_step_20000.pth | 24500 |
| peract2_orbital_handover_dense_chunk8_rel | interm_step_50000.pth | 50000 |
