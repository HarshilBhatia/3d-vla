# grogu status

Updated 2026-10-15 06:34 EDT at bce4ee8.

| job | run | state | node | GPU | progress (this launch) | speed |
|---|---|---|---|---|---|---|
| 4030179 | exp10f_chunk8_pooled_refine | PENDING | - | - |  | - |
| 4030178 | exp10f_chunk8_pooled_refine | RUNNING | grogu-2-25 | A5000 | 65147/75500 | 1.47it/s |

GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded.

| run | newest interm_step | last.pth iter |
|---|---|---|
| peract2_orbital_handover_dense_chunk8_deltam | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine | interm_step_62500.pth | 62500 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_rel | interm_step_60000.pth | 64600 |
| peract2_orbital_handover_dense_chunk8_fullpatch_refine_x0 | interm_step_62500.pth | 62500 |
| peract2_orbital_handover_dense_chunk8_pooled_refine | interm_step_85000.pth | 89600 |
| peract2_orbital_handover_dense_chunk8_rel | interm_step_50000.pth | 50000 |
| peract2_orbital_handover_dense_chunk8_rel_propoffset | interm_step_50000.pth | 50000 |
