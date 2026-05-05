import sys, os, glob
sys.path.insert(0, ".")
from data.processing.rlbench_utils import CustomUnpickler

root = "/grogu/user/harshilb/orbital_rollouts_mini/open_drawer/G1"
episodes = sorted(glob.glob(os.path.join(root, "episode_*")))

for ep in episodes:
    pkl = os.path.join(ep, "low_dim_obs.pkl")
    if not os.path.exists(pkl):
        continue
    with open(pkl, "rb") as f:
        demo = CustomUnpickler(f).load()
    seed_pos = demo.random_seed[2] if demo.random_seed is not None else "None"
    keys = demo.random_seed[1][:10].tolist() if demo.random_seed is not None else []
    print(f"{os.path.basename(ep):12s}  var={demo.variation_number}  T={len(demo):3d}  seed_pos={seed_pos}  keys={keys}")