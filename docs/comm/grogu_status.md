# grogu status

Updated 2026-10-15 00:53 EDT at 94e8973.

| job | run | state | node | GPU | progress (this launch) | speed |
|---|---|---|---|---|---|---|
| 4030292 | exp10b_chunk8_fullpatch_refine | RUNNING | grogu-2-30 | A5000 | 12834/15700 | 1.10it/s |
| 4030294 | exp10c_chunk8_fullpatch_refine_x0 | RUNNING | grogu-2-30 | A5000 | 13264/14800 | 1.15it/s |
| 4030177 | exp10d_chunk8_fullpatch_refine_rel | PENDING | - | - |  | - |
| 4030176 | exp10d_chunk8_fullpatch_refine_rel | RUNNING | grogu-2-25 | A5000 | 29899/73900 | 1.19it/s |
| 4030179 | exp10f_chunk8_pooled_refine | PENDING | - | - |  | - |
| 4030178 | exp10f_chunk8_pooled_refine | RUNNING | grogu-2-25 | A5000 | 35799/75500 | 1.54it/s |

GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded.

| run | newest interm_step | last.pth iter |
|---|---|---|
| peract2_orbital_handover_dense_chunk8_deltam | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine | interm_step_57500.pth | 59600 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_rel | interm_step_55000.pth | 56000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_x0 | interm_step_60000.pth | 60900 |
| peract2_orbital_handover_dense_chunk8_pooled_refine | interm_step_60000.pth | 60300 |
| peract2_orbital_handover_dense_chunk8_rel | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_rel_propoffset | - | 2280 |
