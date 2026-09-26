

3 models (full dataset results)


1. default_3dfa - trained on clean data
2. fixmed_rn - trained on miscal data + added noise 
3. deltaM-EEF - our method, trained on miscal data + added noise.

### TEST LAB EVAL

NOISY LAB EVALUATION - new camera + new miscal.

| Task | default_3dfa | fixmed_rn | deltaM_EEF |
|------|:---:|:---:|:---:|
| insert_onto_square_peg | 0.020 | 0.000 | 0.000 |
| light_bulb_in | 0.000 | 0.010 | 0.000* |
| place_cups | 0.025* | 0.000 | 0.000* |
| place_wine_at_rack_location | 0.092* | 0.050 | 0.000 |
| put_groceries_in_cupboard | 0.060 | 0.000* | 0.007* |
| put_item_in_drawer | 0.000 | 0.000 | 0.000* |
| reach_and_drag | 0.180 | 0.000 | 0.000 |
| stack_blocks | 0.030 | 0.010 | 0.000 |
| stack_cups | 0.000 | 0.000 | 0.000 |
| open_drawer | 0.000 | 0.390 | 0.600 |
| meat_off_grill | 0.750 | 0.070 | 0.000 |
| put_money_in_safe | 0.520 | 0.570 | 0.740 |
| slide_block_to_color_target | 0.510 | 0.480 | 0.500 |
| sweep_to_dustpan_of_size | 0.400 | 0.670 | 0.710 |
| turn_tap | 0.660 | 0.730 | 0.860 |


**Except 5 EASY TASKS, all get 0% performance.**

Now, there are 2 things at play here, new camera and new miscal.


### EVALUATING 3DFA BASE USING CAMERA SCENE DURING TRAINING


| Task (group) | Train SR |
|---|:---:|
| turn_tap (G1) | **1.000** |
| meat_off_grill (G1) | **0.750** |
| slide_block_to_color_target (G2) | **0.600** |
| put_money_in_safe (G1) | **0.550** |
| sweep_to_dustpan_of_size (G1) | **0.350** |
| reach_and_drag (G1) | **0.300** |
| stack_blocks (G3) | **0.200** |
| put_groceries_in_cupboard (G4) | **0.150** |
| place_cups (G1) | **0.100** |
| place_wine_at_rack_location (G2) | **0.050** |
| insert_onto_square_peg (G3) | 0.000 |
| light_bulb_in (G4) | 0.000 |
| stack_cups (G4) | 0.000 |
| open_drawer (G1) | * |
| put_item_in_drawer (G1) | * |


The * ones have not completed running yet. But the tasks that fail above are the ones that get really bad performance - even on TRAIN SET without any miscal. This is not about generalisibility to new views, even using cameras it was trained on, we are getting really bad results for complex tasks. 

The data was made using the de-facto 100 rollouts per task. ( 20 eval rollouts.)

The training curves look good (train and val.)


### PERFORMANCE DEG WITH NOISE 

Evaluating performance degradation with noise. This is on **TURN TAP** which is the easiest task.


| Noise | default_3dfa | fixmed_rn | deltaM_EEF |
|------:|:---:|:---:|:---:|
| 2deg + 2cm| 0.907* | 0.790 | **0.860** |
| 5deg + 5cm| 0.429* | 0.770 | **0.860** |
| 10deg + 10 cm| 0.132* | 0.770 | 0.793* |
| 15deg + 15 cm| 0.000* | 0.710 | **0.770** |
