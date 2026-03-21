
## extrinsics vary

### Goal
check how extrinsics vary across the rollouts and labs.

### Method:
1. add all -- left and right EXTERNAL CAMERA the extrinsic values to a dictoionary, only keep unique values ( within error tolerance of 0.01). Keep a counter of occurance of each extrinsic value. (and this should be per-lab) -- save this, and add loading this in case we change how to plot.

2. now, make 3D plots plotting these values
a) make N, plots 1 per each lab, with their extrinsics plotted. (both cameras). Also color grade them such that we know the frequency of each extrinsic
b) make 1 master plot, with all of these values together.