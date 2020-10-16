# Copyright (c) Open-MMLab. All rights reserved.
from .da_hook import DAHOOKS, DAHook


@DAHOOKS.register_module()
class DADistSamplerSeedHook(DAHook):

    def before_epoch(self, runner):
        if hasattr(runner.data_loader_s.sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.data_loader_s.sampler.set_epoch(runner.epoch)
        elif hasattr(runner.data_loader_s.batch_sampler.sampler, 'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.data_loader_s.batch_sampler.sampler.set_epoch(runner.epoch)

        if hasattr(runner.data_loader_t.sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.data_loader_t.sampler.set_epoch(runner.epoch)
        elif hasattr(runner.data_loader_t.batch_sampler.sampler, 'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.data_loader_t.batch_sampler.sampler.set_epoch(runner.epoch)

