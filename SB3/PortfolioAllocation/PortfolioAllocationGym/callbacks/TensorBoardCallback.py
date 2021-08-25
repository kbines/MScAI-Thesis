from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        self.logger.record('Portfolio Value', env.portfolio_value)

        return True
