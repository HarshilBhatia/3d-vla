# grogu status

Updated 2026-10-14 17:41 EDT at 7441d88.

| job | run | state | node | GPU | progress (this launch) | speed |
|---|---|---|---|---|---|---|
| 4030292 | exp10b_chunk8_fullpatch_refine | PENDING | - | - |  | - |
| 4030291 | exp10b_chunk8_fullpatch_refine | RUNNING | grogu-2-30 | A5000 | 8249/38200 | 1.10it/s |
| 4030294 | exp10c_chunk8_fullpatch_refine_x0 | PENDING | - | - |  | - |
| 4030293 | exp10c_chunk8_fullpatch_refine_x0 | RUNNING | grogu-2-30 | A5000 | 8513/38200 | 1.14it/s |
| 4030177 | exp10d_chunk8_fullpatch_refine_rel | PENDING | - | - |  | - |
| 4030176 | exp10d_chunk8_fullpatch_refine_rel | RUNNING | grogu-2-25 | A5000 | 1127/73900 | 1.19it/s |
| 4030179 | exp10f_chunk8_pooled_refine | PENDING | - | - |  | - |
| 4030178 | exp10f_chunk8_pooled_refine | RUNNING | grogu-2-25 | A5000 | 1365/75500 | 1.52it/s |

GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded.

| run | newest interm_step | last.pth iter |
|---|---|---|
| peract2_orbital_handover_dense_chunk8_deltam | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine | interm_step_32500.pth | 32500 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_rel | interm_step_25000.pth | 27200 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_x0 | interm_step_32500.pth | 32800 |
| peract2_orbital_handover_dense_chunk8_pooled_refine | interm_step_25000.pth | 25800 |
| peract2_orbital_handover_dense_chunk8_rel | interm_step_50000.pth | 50000 |
