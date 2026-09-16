"""Base data preprocessing contract (owned by the data domain)."""


class DataPreprocessor:

    def __init__(self, keypose_only=False, visual_num_history=1, proprio_num_history=None,
                 custom_imsize=None, depth2cloud=None):
        self.keypose_only = keypose_only
        self.visual_num_history = visual_num_history
        self.proprio_num_history = visual_num_history if proprio_num_history is None else proprio_num_history
        self.custom_imsize = custom_imsize
        self.depth2cloud = depth2cloud

    def process_actions(self, actions):
        """Action shape: (B, T, nhand, 3+rot+1)."""
        actions = actions.cuda(non_blocking=True)
        if self.keypose_only:
            actions = actions[:, [-1]]
        return actions

    def process_proprio(self, proprio):
        """Proprio shape: (B, nhist, nhand, 3+rot+1)."""
        proprio = proprio.cuda(non_blocking=True)
        nhist_ = proprio.size(1)
        assert nhist_ >= self.proprio_num_history, "not enough proprio timesteps"
        proprio = proprio[:, :max(self.proprio_num_history, 1)]
        return proprio

    def process_obs(self, rgbs, pcds):
        pass
