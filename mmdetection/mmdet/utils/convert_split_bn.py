import torch
import torch.nn as nn

class SplitBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_splits=2):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_splits - 1)])

    def forward(self, input: torch.Tensor, domain):
        if self.training:  # aux BN only relevant while training
            if domain == 0:
                return super().forward(input)
            elif domain == 1:
                x = []
                for i, a in enumerate(self.aux_bn):
                    x.append(a(input))
                return x[0]
        else:
            out = []
            for i, a in enumerate(self.aux_bn):
                out.append(a(input))
            return torch.cat(out,dim=0)
            #return super().forward(input)

class SplitGroupNorm(torch.nn.GroupNorm):

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, num_splits=2):
        super().__init__(num_groups, num_channels, eps, affine)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([
            nn.GroupNorm(num_groups, num_channels, eps, affine) for _ in range(num_splits - 1)])

    def forward(self, input: torch.Tensor, domain):
        if self.training:  # aux BN only relevant while training
            if domain == 0:
                return super().forward(input)
            elif domain == 1:
                x = []
                for i, a in enumerate(self.aux_bn):
                    x.append(a(input))
                return x[0]
        else:
            out = []
            for i, a in enumerate(self.aux_bn):
                out.append(a(input))
            return torch.cat(out,dim=0)

def convert_splitbn_model(module, num_splits=2):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.
    Args:
        module (torch.nn.Module): input module
        num_splits: number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = timm.models.convert_splitbn_model(model, num_splits=2)
    """
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SplitBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats, num_splits=num_splits)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        for aux in mod.aux_bn:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            aux.num_batches_tracked = module.num_batches_tracked.clone()
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    elif isinstance(module, torch.nn.modules.GroupNorm):
        mod = SplitGroupNorm(
            module.num_groups, module.num_channels, module.eps, module.affine, num_splits=num_splits)
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        for aux in mod.aux_bn:
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_splitbn_model(child, num_splits=num_splits))
    del module
    return mod
