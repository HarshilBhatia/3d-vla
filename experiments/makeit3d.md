What to implement:

1. The action head also takes xyz as new inputs ( with depth!) and extrinsics / instrinsics. 
2. for each token vision, we can get a 3d position ( how we do this is, take avg of all points in that patch). 
3.  implement 3d rope for the each token, this is done over the current positional encodings.

Notes
1. Assume actions tokens to have 0,0,0 positions? Can we do rope on noisy positions ? 


Q:
1. how are action tokens encoded in the model ? 
