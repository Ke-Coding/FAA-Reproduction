from FastAutoAugment.networks import get_model
from summary import summary

if __name__ == '__main__':
    # model = {
    #     'type': 'pyramid',
    #     'depth': 272,
    #     'alpha': 200,
    #     'bottleneck': True,
    # }   # Flops:  4601.980 * 10^6 (4388.790M), Params:  26.211 * 10^6 ( 24.997M)
    
    model = {
        # 'type': 'shakeshake26_2x96d',   # Flops:  3788.611 * 10^6 (3613.101M), Params:  26.778 * 10^6 ( 25.537M)
        # 'type': 'wresnet28_10',       # Flops:  5964.805 * 10^6 (5688.481M), Params:  36.489 * 10^6 ( 34.799M)
        # 'type': 'wresnet40_2',        # Flops:  359.861 * 10^6 (343.190M), Params:   2.246 * 10^6 (  2.142M)
    }
    net = get_model(model)
    summary(net, (3, 32, 32))

