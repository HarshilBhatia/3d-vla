"""Single code path for turning a config into model constructor kwargs.

The trainer used to hand-maintain a ``dict(backbone=..., ...)`` with ~20 entries
copied from the config, while the eval loader filtered ``vars(args)`` through
``inspect.signature``. Any flag added to the config and the constructor but
forgotten in the trainer's dict trained with the constructor default while eval
used the config value -- a silent train/eval divergence (``image_space_sampling``
cost ~30 eval points before it was found).

Both paths now call :func:`build_model_kwargs`. The trainer additionally calls
:func:`assert_model_kwargs_complete` at startup so the next dropped flag is a
crash at step 0 instead of a silent accuracy regression.
"""

import inspect


class _Derived:
    """A constructor param whose value is derived from one or more config keys."""

    def __init__(self, sources, fn, doc):
        self.sources = tuple(sources)
        self.fn = fn
        self.doc = doc

    def __call__(self, cfg):
        return self.fn(cfg)


# Constructor params whose name differs from the config key, or that are derived
# from config values. Everything not listed here is forwarded by exact name.
MODEL_KWARG_MAP = {
    "nhist": _Derived(("num_history",), lambda c: c["num_history"], "config key is num_history"),
    "nhand": _Derived(
        ("bimanual",), lambda c: 2 if c["bimanual"] else 1, "2 hands when bimanual else 1"
    ),
    "relative": _Derived(
        ("relative_action",), lambda c: c["relative_action"], "config key is relative_action"
    ),
}

# Constructor params that legitimately have no config key and are expected to
# take their constructor default. Anything not listed raises in strict mode, so
# adding a constructor param forces a decision about how it is configured.
ALLOWED_CONSTRUCTOR_DEFAULTS = frozenset({
    # denoise2d-only legacy RoPE-ΔM knobs, never exposed in config/config.yaml
    "use_rope_delta_m",
    "rope_lambda_reg",
})

# Config keys that share a name with a constructor param but must not be
# forwarded. Empty by design: if it ever needs an entry, that entry is the
# documentation for why a config value is deliberately withheld from the model.
WITHHELD_CONFIG_KEYS = frozenset()


def _as_dict(args):
    if isinstance(args, dict):
        return dict(args)
    return dict(vars(args))


def model_signature_params(model_class):
    """Constructor parameter names of ``model_class``, excluding ``self``."""
    params = inspect.signature(model_class.__init__).parameters
    return [
        name
        for name, p in params.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def build_model_kwargs(args, model_class):
    """Build constructor kwargs for ``model_class`` from a config/args object.

    Every constructor parameter is resolved from, in order:

    1. :data:`MODEL_KWARG_MAP` (explicit renames and derivations), then
    2. an identically-named config key, then
    3. left out, so the constructor default applies.

    Args:
        args: config object (``SimpleNamespace``/``Namespace``) or plain dict.
        model_class: the model class whose ``__init__`` signature is the filter.

    Returns:
        dict of constructor kwargs.
    """
    cfg = _as_dict(args)
    sig = model_signature_params(model_class)

    kwargs = {}
    for name in sig:
        mapping = MODEL_KWARG_MAP.get(name)
        if mapping is not None:
            missing = [s for s in mapping.sources if s not in cfg]
            if missing:
                raise KeyError(
                    f"model kwarg '{name}' ({mapping.doc}) needs config key(s) {missing}, "
                    f"which are absent from the config"
                )
            kwargs[name] = mapping(cfg)
        elif name in cfg and name not in WITHHELD_CONFIG_KEYS:
            kwargs[name] = cfg[name]
    return kwargs


def assert_model_kwargs_complete(args, model_class, kwargs=None):
    """Raise if any config value that the model could consume is being dropped.

    Two directions are checked:

    * a config key matching a constructor param name that did not make it into
      ``kwargs`` (the ``image_space_sampling`` failure mode), and
    * a constructor param that is absent from ``kwargs`` and not in
      :data:`ALLOWED_CONSTRUCTOR_DEFAULTS`, i.e. silently taking its default.

    Raises:
        ValueError: on either condition, listing every offending name.
    """
    cfg = _as_dict(args)
    sig = model_signature_params(model_class)
    if kwargs is None:
        kwargs = build_model_kwargs(args, model_class)

    dropped = sorted(
        name
        for name in sig
        if name in cfg and name not in kwargs and name not in WITHHELD_CONFIG_KEYS
    )
    defaulted = sorted(
        name
        for name in sig
        if name not in kwargs and name not in ALLOWED_CONSTRUCTOR_DEFAULTS
    )

    problems = []
    if dropped:
        problems.append(
            f"config keys match constructor params of {model_class.__name__} but are not "
            f"forwarded (would silently use constructor defaults): {dropped}"
        )
    if defaulted:
        problems.append(
            f"constructor params of {model_class.__name__} have no config key and are not "
            f"in ALLOWED_CONSTRUCTOR_DEFAULTS: {defaulted}. Add them to config/config.yaml "
            f"(preferred) or to ALLOWED_CONSTRUCTOR_DEFAULTS with a comment."
        )
    if problems:
        raise ValueError(
            "model construction would drop configuration:\n  - " + "\n  - ".join(problems)
        )

    unmapped = sorted(
        name
        for name, mapping in MODEL_KWARG_MAP.items()
        if name in sig and name not in kwargs
    )
    if unmapped:
        raise ValueError(
            f"explicit MODEL_KWARG_MAP entries were not consumed for "
            f"{model_class.__name__}: {unmapped}"
        )
