
## TASK 
i want to now generate rollouts ( RL bench-peract)

### HOW
we have groups (orbital-cameras_grouped). [G1 - G6].
For each task, create mapping of 3 camera groups. Make sure task-grouping mapping is uniform. ( each Group has 6 tasks).
For each task, i want to generate 90 rollouts, 30 per-camera group.
All should be saved in a zarr file (how RLbench already does this).


### DEBUG 
Before generating large scale data, lets also generate video of the rollouts, for each task - per camera. So 3 videos per task, and total 18 * 3 = 54 videos.

