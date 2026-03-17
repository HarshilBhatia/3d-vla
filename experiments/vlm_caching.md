Now the full path in cached mode is:
  1. get_shard() skips video decode, strips "video" from modality configs → content.images = {}
  2. Processor sees content.images is empty → skips _get_vlm_inputs entirely → returns only {state, action,
  action_mask, embodiment_id}
  3. Collator sees cache_global_idx → loads backbone features from shard cache, skips _BACKBONE_INPUT_KEYS
  from the raw datapoint
