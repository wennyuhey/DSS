# Copyright (c) Open-MMLab. All rights reserved.
import time

from .da_hook import DAHOOKS, DAHook


@DAHOOKS.register_module()
class DAIterTimerHook(DAHook):

    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()
