import torch.optim as optim

def SGDoptimizer(net, lr, optimizer_args):
    momentum = optimizer_args["momentum"]
    return optim.SGD(net.parameters(), lr, momentum)