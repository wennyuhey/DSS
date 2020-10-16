# Copyright (c) Open-MMLab. All rights reserved.
from .checkpoint import CheckpointHook
from .closure import ClosureHook
from .ema import EMAHook
from .hook import HOOKS, Hook
from .iter_timer import IterTimerHook
from .logger import (LoggerHook, MlflowLoggerHook, PaviLoggerHook,
                     TensorboardLoggerHook, TextLoggerHook, WandbLoggerHook)
from .lr_updater import LrUpdaterHook
from .memory import EmptyCacheHook
from .momentum_updater import MomentumUpdaterHook
from .optimizer import Fp16OptimizerHook, OptimizerHook
from .sampler_seed import DistSamplerSeedHook
from .sync_buffer import SyncBuffersHook

from .da_checkpoint import DACheckpointHook
from .da_closure import DAClosureHook
from .da_ema import DAEMAHook
from .da_hook import DAHOOKS, DAHook
from .da_iter_timer import DAIterTimerHook
from .logger import (DALoggerHook, DAMlflowLoggerHook, DAPaviLoggerHook,
                       DATensorboardLoggerHook, DATextLoggerHook, DAWandbLoggerHook)
from .da_lr_updater import DALrUpdaterHook
from .da_memory import DAEmptyCacheHook
from .da_momentum_updater import DAMomentumUpdaterHook
from .da_optimizer import DAFp16OptimizerHook, DAOptimizerHook
from .da_sampler_seed import DADistSamplerSeedHook
from .da_sync_buffer import DASyncBuffersHook


__all__ = [
    'HOOKS', 'Hook', 'CheckpointHook', 'ClosureHook', 'LrUpdaterHook',
    'OptimizerHook', 'Fp16OptimizerHook', 'IterTimerHook',
    'DistSamplerSeedHook', 'EmptyCacheHook', 'LoggerHook', 'MlflowLoggerHook',
    'PaviLoggerHook', 'TextLoggerHook', 'TensorboardLoggerHook',
    'WandbLoggerHook', 'MomentumUpdaterHook', 'SyncBuffersHook', 'EMAHook',
    'DAHOOKS', 'DAHook', 'DACheckpointHook', 'DAClosureHook', 'DALrUpdaterHook',
    'DAOptimizerHook', 'DAFp16OptimizerHook', 'DAIterTimerHook',
    'DADistSamplerSeedHook', 'DAEmptyCacheHook', 'DALoggerHook', 'DAMlflowLoggerHook',
    'DAPaviLoggerHook', 'DATextLoggerHook', 'DATensorboardLoggerHook',
    'DAWandbLoggerHook', 'DAMomentumUpdaterHook', 'DASyncBuffersHook', 'DAEMAHook'

]
