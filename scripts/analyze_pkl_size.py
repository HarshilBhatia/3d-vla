"""Analyze the in-memory size breakdown of a pickle file."""
import pickle
import sys
import numpy as np


def sizeof_fmt(num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num) < 1024.0:
            return f'{num:.2f} {unit}'
        num /= 1024.0
    return f'{num:.2f} TB'


def analyze(obj, name='root', depth=0, max_depth=5):
    indent = '  ' * depth
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
        print(f'{indent}{name}: ndarray {obj.shape} dtype={obj.dtype} => {sizeof_fmt(size)}')
        return size
    elif isinstance(obj, dict):
        print(f'{indent}{name}: dict ({len(obj)} keys)')
        total = 0
        if depth < max_depth:
            for k, v in obj.items():
                total += analyze(v, str(k), depth + 1, max_depth)
        return total
    elif isinstance(obj, (list, tuple)):
        tname = type(obj).__name__
        total = 0
        if depth < max_depth:
            if len(obj) <= 5:
                print(f'{indent}{name}: {tname} (len={len(obj)})')
                for i, v in enumerate(obj):
                    total += analyze(v, f'[{i}]', depth + 1, max_depth)
            else:
                print(f'{indent}{name}: {tname} (len={len(obj)}) — sampling first/last')
                s0 = analyze(obj[0], '[0] sample', depth + 1, max_depth)
                analyze(obj[-1], '[-1] sample', depth + 1, max_depth)
                total = s0 * len(obj)
        else:
            print(f'{indent}{name}: {tname} (len={len(obj)})')
        return total
    else:
        d = getattr(obj, '__dict__', None)
        if d and depth < max_depth:
            print(f'{indent}{name}: {type(obj).__name__} (object, {len(d)} attrs)')
            total = 0
            for k, v in d.items():
                total += analyze(v, k, depth + 1, max_depth)
            return total
        size = sys.getsizeof(obj)
        print(f'{indent}{name}: {type(obj).__name__} => {sizeof_fmt(size)}')
        return size


path = sys.argv[1] if len(sys.argv) > 1 else \
    '/grogu/user/harshilb/orbital_rollouts/open_drawer/G1/episode_0/low_dim_obs.pkl'

print(f'Loading {path} ...')
with open(path, 'rb') as f:
    data = pickle.load(f)

print(f'Top-level type: {type(data).__name__}\n')
total = analyze(data, 'data', max_depth=5)
print(f'\nEstimated total in-memory: {sizeof_fmt(total)}')
