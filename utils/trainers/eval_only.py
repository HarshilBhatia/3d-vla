"""TrainTester that only runs evaluation (no training)."""

from .base import BaseTrainTester


class EvalOnlyTrainTester(BaseTrainTester):
    """TrainTester that only runs evaluation (no training)."""

    def __init__(self, args, dataset_cls, model_cls):
        args.eval_only = True
        super().__init__(args, dataset_cls, model_cls)

    def main(self):
        """Run evaluation only."""
        return super().main()
