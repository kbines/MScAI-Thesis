from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        self.logger.record('Cum. Returns', env.cumulative_returns[-1])
        self.logger.record('Sharpe', env.sharpe)
        self.logger.record('Sortino', env.sortino)
        self.logger.record('Calmar', env.calmar)
        self.logger.record('PSR', env.psr)

        return True
