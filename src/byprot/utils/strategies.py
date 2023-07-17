from typing import Dict, List, Union

from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.strategies.sharded import DDPShardedStrategy
from torch.optim import Optimizer

from pytorch_lightning.utilities.imports import _FAIRSCALE_AVAILABLE

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS

    class DDPShardedFBOStrategy(DDPShardedStrategy):
        strategy_name = 'ddp_sharded_fbo'

        def __init__(self, force_broadcast_object=True, **kwargs) -> None:
            super().__init__(**kwargs)
            self.force_broadcast_object = force_broadcast_object

        def _wrap_optimizers(self, optimizers: List[Optimizer]) -> List["OSS"]:
            oos_optimizers = super()._wrap_optimizers(optimizers)
            if self.force_broadcast_object:
                for oos_optimizer in oos_optimizers:
                    oos_optimizer.force_broadcast_object = True
            return oos_optimizers

        @classmethod
        def register_strategies(cls, strategy_registry: Dict) -> None:
            strategy_registry.register(
                cls.strategy_name,
                cls,
                description="DDP Shared Strategy with force_broadcast_object enabled",
            )

    StrategyRegistry.register(
        "ddp_sharded_fbo",
        DDPShardedFBOStrategy,
        description="DDP Shared Strategy with force_broadcast_object enabled",
        force_broadcast_object=True,
    )