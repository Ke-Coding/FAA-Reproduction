import torch
import torchsummary
from thop import profile
from tensorboardX import SummaryWriter


__all__ = ['summary']


def summary(net, input_size, using_torchsummary=False):
    
    if using_torchsummary:
        torchsummary.summary(net, input_size)

    x = torch.rand((1, input_size[0], input_size[1], input_size[2]))
    flops, params = profile(net, inputs=(x, ))
    print('Flops:  %7.3f * 10^6 (%7.3fM)' % (flops/1000/1000, flops/1024/1024))
    print('Params: %7.3f * 10^6 (%7.3fM)' % (params/1000/1000, params/1024/1024))
    # print(net(x))

    # with SummaryWriter(comment=type(net).__name__) as w:
    # 	w.add_graph(net, x)	# cd betaNAS/prototype/summary => tensorboard --logdir runs
