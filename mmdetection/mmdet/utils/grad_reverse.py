from torch.autograd.function import Function


class GradReverse(Function):
    @classmethod
    def forward(cls, ctx, x):
        #ctx.save_for_backward(result)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #pdb.set_trace()
        #result, = ctx.saved_tensors
        return (grad_output * (-1))
