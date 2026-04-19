import argparse


def str_none(value):
    if value.lower() in ['none', 'null', 'nil'] or len(value) == 0:
        return None
    else:
        return value


def str2bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def round_floats(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


def _fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def count_parameters(model, depth=1):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    rows = []
    for name, submodule in model.named_modules():
        level = name.count('.')
        if level >= depth or (level == 0 and name == ''):
            continue
        tr = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        fr = sum(p.numel() for p in submodule.parameters() if not p.requires_grad)
        if tr + fr == 0:
            continue
        indent = "  " * level
        rows.append((indent + name, tr, fr))

    name_w = max((len(r[0]) for r in rows), default=4)
    tr_w = max((len(_fmt(r[1])) for r in rows), default=8)

    print(f"\n{'─'*60}")
    print(f"  Model parameters")
    print(f"  Total:     {_fmt(total):>10}   Trainable: {_fmt(trainable):>10}   Frozen: {_fmt(frozen):>10}")
    print(f"{'─'*60}")
    header = f"  {'Module':<{name_w}}   {'Trainable':>{tr_w}}   Frozen"
    print(header)
    print(f"  {'-'*( len(header)-2)}")
    for name, tr, fr in rows:
        print(f"  {name:<{name_w}}   {_fmt(tr):>{tr_w}}   {_fmt(fr)}")
    print(f"{'─'*60}\n")
