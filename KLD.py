import torch
from Model import Net
import copy
from torchvision import datasets, transforms
import torch.nn.functional as F
from ForwardPass import quantForward
from QuantFunc import quantize_tensor, dequantize_tensor
from Stats import gatherStats
from QuantLayer import quantizeConv
import numpy as np
import math


######## Models #######
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()
q_model = copy.deepcopy(model)

######## DataLoaders #######
kwargs = {'num_workers': 0, 'pin_memory': True}

dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

test_loader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=True, **kwargs)

###### Quantized Model #######
bitwidth = [8, [2, 2, 8], [8, 8, 8], [8, 8, 8]]
stats = gatherStats(q_model, test_loader)


dummy = torch.rand([1,1,28,28])
filter_sums_original = np.zeros((model.conv1(dummy).shape[1],len(dataset)))
filter_sums_quantized = np.zeros((model.conv1(dummy).shape[1], len(dataset)))


for i, (data, target) in enumerate(test_loader):
    Conv1out = F.relu(model.conv1(data))
    Conv1out = Conv1out.reshape(Conv1out.detach().numpy().shape[0], Conv1out.detach().numpy().shape[1], -1)
    for filter_it in range(Conv1out.shape[1]):
        filter_sums_original[filter_it][i] = np.mean(Conv1out[0].detach().numpy()[filter_it], axis=0)

    x = quantize_tensor(data, bitwidth[0], min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
    x, scale_next, zero_point_next = quantizeConv(x.tensor, q_model.conv1, stats['conv2'], x.scale,
                                                          x.zero_point,bitwidth[1])
    QConv1out = dequantize_tensor(x, scale_next, zero_point_next)
    QConv1out = QConv1out.reshape(QConv1out.detach().numpy().shape[0], QConv1out.detach().numpy().shape[1], -1)
    for filter_iter in range(QConv1out.shape[1]):
        filter_sums_quantized[filter_iter][i] = np.mean(QConv1out[0].detach().numpy()[filter_iter], axis=0)


mean_original = np.zeros((filter_sums_original.shape[0]))
std_original = np.zeros((filter_sums_original.shape[0]))
for it in range(filter_sums_original.shape[0]):
    mean_original[it] = np.mean(filter_sums_original[it])
    std_original[it] = np.std(filter_sums_original[it])


mean_quant = np.zeros((filter_sums_quantized.shape[0]))
std_quant = np.zeros((filter_sums_quantized.shape[0]))
for it in range(filter_sums_quantized.shape[0]):
    mean_quant[it] = np.mean(filter_sums_quantized[it])
    std_quant[it] = np.std(filter_sums_quantized[it])


KL = np.zeros((filter_sums_original.shape[0]))
for it in range(filter_sums_original.shape[0]):
    KL[it] = math.log(std_quant[it]/std_original[it])+(std_original[it]**2 - std_quant[it]**2 +(mean_original[it]-mean_quant[it])**2)/(2*std_quant[it]**2)
R= np.mean(KL)

print(1)