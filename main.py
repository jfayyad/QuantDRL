from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import namedtuple
import copy
from torchinfo import summary
import numpy as np

# #################################### MNIST Model #################################### #
class Net(nn.Module):
    def __init__(self, mnist=True):

        super(Net, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# #################################### Training #################################### #
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    batch_size = 64
    test_batch_size = 64
    epochs = 5
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = True
    no_cuda = False

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model

# model = main()

####################################################################################################
# #################################### Quantization Functions #################################### #
####################################################################################################

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])
WTensor = namedtuple('WTensor', ['tensor', 'scale', 'zero_point'])
BTensor = namedtuple('BTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val, num_bits):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    return scale, zero_point


def quantize_tensor(x, num_bits, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


# ################# Rework Forward pass of Linear and Conv Layers to support Quantisation ################# #
def quantizeLayer(x, layer, stat, scale_x, zp_x, num_bits):
    # for both conv and linear layers
    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data, num_bits=num_bits[0])
    b = quantize_tensor(layer.bias.data, num_bits=num_bits[1])

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    # This is Quantisation Artihmetic
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits[2])
    # Preparing input by shifting
    X = x.float() - zp_x

    layer.weight.data = scale_x * scale_w * (layer.weight.data - zp_w)
    layer.bias.data = scale_b * (layer.bias.data - zp_b)

    # All int computation
    x = (layer(X) / scale_next) + zero_point_next

    # Perform relu too
    x = F.relu(x)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next


def quantizeLayerConv(x, w, b, layer, stat, scale_x, zp_x, num_bits):
    # for both conv and linear layers
    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    w_tensor = w.tensor.detach().clone()

    # This is Quantisation Artihmetic
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits[2])
    # Preparing input by shifting
    X = x.float() - zp_x

    for i in range(layer.weight.size(dim=0)):
        temp = scale_x * scale_w[i] * (w_tensor[i] - zp_w[i])
        w_tensor[i] = temp

    # layer.weight.data = scale_x * scale_w * (layer.weight.data - zp_w)
    layer.weight.data = w_tensor
    layer.bias.data = scale_b * (layer.bias.data - zp_b)

    # All int computation
    x = (layer(X) / scale_next) + zero_point_next

    # Perform relu too
    x = F.relu(x)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next


# def quantizeLayerConv(x, layer, stat, scale_x, zp_x, num_bits):
#     # for both conv and linear layers
#     # cache old values
#     W = (layer.weight.data).detach().clone()
#     B = (layer.bias.data).detach().clone()
#
#     k = layer.weight.size(dim=0)
#     w_tensor = (layer.weight.data).detach().clone()
#     scale_w = []
#     zp_w = []
#
#     for i in range(k):
#         w_temp = quantize_tensor(layer.weight[i].data, num_bits=num_bits[0][i])
#         w_tensor[i] = w_temp.tensor.float()
#         scale_w.append(w_temp.scale)
#         zp_w.append(w_temp.zero_point)
#
#     b = quantize_tensor(layer.bias.data, num_bits=num_bits[1])
#     layer.bias.data = b.tensor.float()
#     scale_b = b.scale
#     zp_b = b.zero_point
#
#     scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'], num_bits=num_bits[2])
#     # Preparing input by shifting
#     X = x.float() - zp_x
#
#     for i in range(k):
#         temp = scale_x * scale_w[i] * (w_tensor[i] - zp_w[i])
#         w_tensor[i] = temp
#
#     # layer.weight.data = scale_x * scale_w * (layer.weight.data - zp_w)
#
#     layer.weight.data = w_tensor
#     layer.bias.data = scale_b * (layer.bias.data - zp_b)
#
#     w0 = layer.weight[0].data
#     w1 = layer.weight[1].data
#
#     # All int computation
#     x = (layer(X) / scale_next) + zero_point_next
#
#     # Perform relu too
#     x = F.relu(x)
#
#     # Reset weights for next forward pass
#     layer.weight.data = W.detach().clone()
#     layer.bias.data = B.detach().clone()
#
#     return x, scale_next, zero_point_next

# ################ Get Max and Min Stats for Quantizing Activations of Network ################  #
# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] = 10000 #This should be changed to len(dataste)

    return stats


# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = F.relu(model.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = F.relu(model.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    stats = updateStats(x, stats, 'fc1')
    x = F.relu(model.fc1(x))
    stats = updateStats(x, stats, 'fc2')
    x = model.fc2(x)
    return stats


# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)

    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {"max": value["max"] / value["total"], "min": value["min"] / value["total"]}
    return final_stats

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<><<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Forward Pass for Quantised Inference >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

def quantForward(model, x, bitwidth, stats):
    # Quantise before inputting into incoming layers
    x = quantize_tensor(x, bitwidth[0], min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

    if per_ch_conv1 == True:
        # x, scale_next, zero_point_next = quantizeLayerConv(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point, bitwidth_conv[1])
        x, scale_next, zero_point_next = quantizeLayerConv(x.tensor, quantConv_w[0], quantConv_b[0],  model.conv1, stats['conv2'], x.scale, x.zero_point, bitwidth_conv[1])
    else:
        x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point, bitwidth[1])

    x = F.max_pool2d(x, 2, 2)

    if per_ch_conv2 == True:
        # x, scale_next, zero_point_next = quantizeLayerConv(x, model.conv2, stats['fc1'], scale_next, zero_point_next, bitwidth_conv[2])
        x, scale_next, zero_point_next = quantizeLayerConv(x, quantConv_w[1], quantConv_b[1], model.conv2, stats['fc1'], scale_next, zero_point_next, bitwidth_conv[2])
    else:
        x, scale_next, zero_point_next = quantizeLayer(x, model.conv2, stats['fc1'], scale_next, zero_point_next, bitwidth[2])

    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'], scale_next, zero_point_next, bitwidth[3])
    # Back to dequant for final layer
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
    x = model.fc2(x)
    return F.log_softmax(x, dim=1)


# #################################### Testing Function for Quantisation #################################### #
def testQuant(model, test_loader, bitwidth, quant=False, stats=None):
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
                output = quantForward(model, data, bitwidth, stats)
            else:
                output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    Acc = correct / len(test_loader.dataset)
    return Acc

# #################################### Quantize Weights and Biases only ONCE #################################### #


def quantizeConv_WB(layer, num_bits):

    k = layer.weight.size(dim=0)
    w_tensor = (layer.weight.data).detach().clone()
    scale_w = []
    zp_w = []

    # num_bits = [np.random.randint(1, 2, k), 8]

    for i in range(k):
        w_temp = quantize_tensor(layer.weight[i].data, num_bits=num_bits[0][i])
        w_tensor[i] = w_temp.tensor.float()
        scale_w.append(w_temp.scale)
        zp_w.append(w_temp.zero_point)

    # scale_w = torch.FloatTensor(scale_w)
    # zp_w = torch.FloatTensor(zp_w)
    b = quantize_tensor(layer.bias.data, num_bits=num_bits[1])

    return WTensor(tensor=w_tensor, scale=scale_w, zero_point=zp_w), BTensor(tensor=b.tensor, scale=b.scale, zero_point=b.zero_point)

def quant_WB(bits_conv):
    quantConv_w = []
    quantConv_b = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name=='conv1':
                num_bits=bits_conv[0]
            if name=='conv2':
                num_bits=bits_conv[1]
            temp_w, temp_b = quantizeConv_WB(module, num_bits)
            quantConv_w.append(temp_w)
            quantConv_b.append(temp_b)

    return quantConv_w, quantConv_b

# #################################### Get Accuracy of Non Quantised Model #################################### #
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()
q_model = copy.deepcopy(model)

# for param in model.parameters():
#     print(param.nelement())

kwargs = {'num_workers': 0, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=10, shuffle=True, **kwargs)

# testQuant(q_model, test_loader, quant=False)

# #################################### Test Quantised Inference Of Model #################################### #
# # Uniform Quantization
# w = 8
# b = 8
# a = 8
# bitwidth = [a, [w, b, a], [w, b, a], [w, b, a]]

# Custom bitwidth: ['input', 'conv1','conv2','fc1' ] ; e.g., conv1 => weight, bias, activation for next layer (input)
bitwidth = [8, [8, 8, 8], [8, 8, 8], [8, 8, 8]]

per_ch_conv1 = False
per_ch_conv2 = False

b_min_conv1 = 1
b_max_conv1 = 2

b_min_conv2 = 1
b_max_conv2 = 2

k_conv1 = model.conv1.weight.size(dim=0)
k_conv2 = model.conv2.weight.size(dim=0)

bits_conv1_w = np.random.randint(b_min_conv1, b_max_conv1+1, k_conv1)
bits_conv2_w = np.random.randint(b_min_conv2, b_max_conv2+1, k_conv2)

bitwidth_conv = [bitwidth[0],
                 [bits_conv1_w, bitwidth[1][1], bitwidth[1][2]],
                 [bits_conv2_w, bitwidth[2][1], bitwidth[2][2]],
                 [bitwidth[3][0], bitwidth[3][1], bitwidth[3][2]]
                 ]

stats = gatherStats(q_model, test_loader)
# # print(stats)

quantConv_w, quantConv_b = quant_WB(bits_conv=[[bits_conv1_w, bitwidth[1][1],], [bits_conv2_w, bitwidth[2][1]]])

acc = testQuant(q_model, test_loader, bitwidth, quant=True, stats=stats)

# summary(model, input_size=(1, 1, 28, 28))
# summary(q_model, input_size=(1, 1, 28, 28))