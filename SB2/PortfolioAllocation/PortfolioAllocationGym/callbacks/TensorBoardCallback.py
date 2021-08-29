from stable_baselines.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings(action='ignore',
                        category=DeprecationWarning,
                        module='stable_baselines')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

class TensorBoardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        self.logger.record('Portfolio Value', env.portfolio_value)

        return True
