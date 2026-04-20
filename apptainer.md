apptainer shell --nv --fakeroot --writable --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run  --bind /grogu/user/harshilb/train_logs:/grogu/user/harshilb/train_logs --env PATH=/root/miniconda3/envs/3dfa/bin:$PATH  my_eval_env




apptainer shell --nv --fakeroot --writable --bind /usr/bin/xvfb-run:/usr/bin/xvfb-run  --bind /grogu/user/harshilb/:/grogu/user/harshilb/ --env PATH=/root/miniconda3/envs/3dfa/bin:$PATH  my_eval_env
