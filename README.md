
Install PyRep

For PerAct2:
```
> git clone https://github.com/markusgrotz/PyRep.git
```

For PerAct and HiveFormer:
```
> git clone https://github.com/stepjam/PyRep.git
```

The rest of the steps are the same for all PyRep versions:
```
> cd PyRep/
> wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
> tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz;
> echo "export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> $HOME/.bashrc; 
> echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> $HOME/.bashrc;
> echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> $HOME/.bashrc;
> source $HOME/.bashrc;
> conda activate 3dfa
> pip install -r requirements.txt; pip install -e .; cd ..
```

Next step is to install RLBench.

For PerAct2:
```
> git clone https://github.com/markusgrotz/RLBench.git
> cd RLBench; pip install -r requirements.txt; pip install -e .; cd ..;
```

For PerAct:
```
> git clone https://github.com/MohitShridhar/RLBench.git
> cd RLBench; git checkout -b peract --track origin/peract; pip install -r requirements.txt; pip install -e .; cd ..;
```
Remember to modify the success condition of `close_jar` task in RLBench, as the original condition is incorrect.  See this [pull request](https://github.com/MohitShridhar/RLBench/pull/1) for more detail.

For HiveFormer:
```
> git clone https://github.com/rjgpinel/RLBench.git
> cd RLBench; pip install -r requirements.txt; pip install -e .; cd ..;
```


## Model API
Training:
```
instr = tokenizer(instr).cuda(non_blocking=True)
out = 3dfa(
    gt_action, None, rgbs, None, pcds, instr, proprio,
    run_inference=True
)
```
Inference:
```
instr = tokenizer(instr).cuda(non_blocking=True)
out = 3dfa(
    None, action_mask, rgbs, None, pcds, instr, proprio,
    run_inference=True
)
```
where:
```
- gt_action: float tensor of shape (bs, num_trajectory_steps, num_arms, 3+R+1), where R=4 for quaternion (converted internally to 6D representation), R=3 for Euler 
- action_mask: torch.zeros(bs, num_trajectory_steps, num_arms).bool()
- rgbs: float tensor of shape (bs, num_cameras, 3, H, W)
- pcds: float tensor of shape (bs, num_cameras, 3, H, W)
- instr: raw text, converted to tensor by the tokenizer
- proprio: float tensor of shape (bs, num_trajectory_steps, num_arms, 3+R)
- out: float tensor (bs, num_trajectory_steps, num_arms, 3+R+1) if run_inference is True else loss
```


## Data Preparation
We use zarr to package the training/validation data for faster loading (see the discussion in the appendix of our paper).

For the test episodes, we do NOT need rendered images and other observations, we only need the random states to create the scene and place the objects. Not loading the pre-rendered demos dramatically speeds up the evaluation process.

The reason previous works require rendering the episodes is because they rely on an inefficient version of the function `get_stored_demos`. We optimized that and provide the clean, observation-free test data for convenience.

### PerAct2
Download pre-packaged data and test seeds using:
```
> bash scripts/rlbench/peract2_datagen_fast.sh
```
or package the data yourself (not recommended) by running:
```
> bash scripts/rlbench/peract2_datagen.sh
```

### PerAct
For PerAct, we convert 3DDA's .dat files to zarr. The following script will download the 3DDA data and repackage them. It will also download the clean test seeds for PerAct:
```
> bash scripts/rlbench/peract_datagen.sh
```

### HiveFormer 74 tasks
Since there are a lot of tasks for HiveFormer, we show how to generate the raw data from scratch and then package it. We do NOT generate test seeds - instead the seeds/internal states are automatically generated while testing. We showcase the pipeline for the task close_door. You can similarly generate the data for all other tasks. You do NOT need to generate data for multiple tasks to train a model, this is a single-task setup.
```
> bash scripts/rlbench/hiveformer_datagen.sh
```


## PerAct2 checkpoint
You can download a trained checkpoint [here](https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/3dfa_peract2.pth).

## Training
```
> bash scripts/rlbench/train_peract2.sh
> bash scripts/rlbench/train_peract.sh
> bash scripts/rlbench/train_hiveformer.sh
```
Make sure to modify the path to the dataset!

To run 3DDA, you can simply change the ```denoise_model``` to ```ddpm```. Note that this will still use 3DFA's design choices though, such as density-biased sampling.

## Online evaluation
```
> bash online_evaluation_rlbench/eval_peract2.sh
> bash online_evaluation_rlbench/eval_peract.sh
> bash online_evaluation_rlbench/eval_hiveformer.sh
```
NOTE: you need to change the paths to the test dataset and your checkpoint!

## EXEC


apptainer exec --nv my_eval_env/ /bin/bash -c "
    source /home/harshilb/miniconda3/etc/profile.d/conda.sh && \
    conda activate /home/harshilb/miniconda3/envs/3dfa && \
    python your_eval_script.py"

  apptainer shell --env PATH=/root/miniconda3/envs/3dfa/bin:$PATH your.sif                                                                                