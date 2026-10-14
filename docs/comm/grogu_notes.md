# Notes from grogu

- 2026-10-14 14:35 EDT: **all six exp 10 runs are now training on grogu**, including **10f (pooled_refine, job 4030178 on grogu-4-8, started 14:17, resumed from 16.7k)**. Do not launch 10f on Delta while it runs here, or the same run trains twice. The user decides which site keeps it; grogu will cancel its copy only on the user's word.
- val.zarr on Delta: the re-push (`DELTA_USER=hbhatia1 scripts/data/push_to_delta.sh handover_chunk8`) needs the user's Duo approval on grogu; it has not run yet.
- Batch per GPU: 10a, 10e at 64 (global 128, to 50k); 10b, 10c, 10d, 10f at 32 (global 64, to 100k). A5000, A6000 and RTX 3090 run at about the same speed.
