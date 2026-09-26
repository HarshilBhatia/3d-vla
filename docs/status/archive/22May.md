
Things to do 
1. Collect data.


1. look at graphs( transfer data), start online eval. (1 run crashed - great). 
resume the run, but eval anyways. 
2. Start pi training. -- 
extracting zip. Done
3. write a bit of the method section. 


how to benchmark pi_0.5, like miscalibration? / what about new camera pose? (because if i add new pose it fails.)


slightly confused which is best for main eval, a) or b) or do both? i think we discussed b, but b has 2 contributing factors.
There are 2 axis of evaluation, 
a) keeping camera fixed (rendering same - as the training set.) + new miscalibration
b) having a new camera + with new miscalibration. having a new\ test time camera also degrades performance, but this is not really our contribution ? ( the only difference that previous methods is, we have varying cameras (the lab concept) = which improve performance.)


Things that is not clear: 
\deltaM is being premultiplied to the 3D RoPE tokens
3D is not clear.
maybe shouldn't call it camera token.
input - reproject images, make miscalibrated 3D point cloud.


run the folloiwing:
light_bulb_in: G4, place_cups: G1, stack_cups: G4. on the camer

okay so some results:
1. light_bulb_in: G4, place_cups: G1, stack_cups: G4, on deltaM (trained with noise) - with the miscal they were trained on 0% success rate.

do the follwoing: 
1. run peract data collection for 2-3 tasks (single setup)
