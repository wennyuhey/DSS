# Copyright (c) Open-MMLab. All rights reserved.
from .da_hook import DAHOOKS, DAHook


@DAHOOKS.register_module()
class DAClosureHook(DAHook):

    def __init__(self, fn_name, fn):
        assert hasattr(self, fn_name)
        assert callable(fn)
        setattr(self, fn_name, fn)
