import torch
from Model import Net
import copy
from torchvision import datasets, transforms
import torch.nn.functional as F
from ForwardPass import quantForward
from QuantFunc import quantize_tensor, dequantize_tensor
from Stats import gatherStats
from QuantLayer import quantizeConv, quantizeLayer
import numpy as np
import math


######## Models #######
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()
q_model = copy.deepcopy(model)

######## DataLoaders #######
kwargs = {'num_workers': 0, 'pin_memory': True}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.MNIST = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    def __getitem__(self, index):
        data, target = self.MNIST[index]

        return data, target, index

    def __len__(self):
        return len(self.MNIST)

dataset = MyDataset()

test_loader = torch.utils.data.DataLoader(dataset,batch_size=100, shuffle=False, **kwargs)

###### Quantized Model #######
bitwidth = [8, [8, 8, 8], [8, 8, 8], [8, 8, 8]]

stats = gatherStats(q_model, test_loader)




correct_original = 0
correct_quantized = 0

def get_conv_act(layer,idx):
    filter_size = layer.shape[1]
    act_layer = np.zeros((filter_size, 10000))
    layer = layer.reshape(layer.detach().numpy().shape[0], layer.detach().numpy().shape[1], -1)
    for filter_it in range(layer.shape[1]):
        for index in range(idx.shape[0]):
            act_layer[filter_it][index + (i * (idx.shape[0]))] = np.mean(
                layer[index].detach().numpy()[filter_it], axis=0)
    return act_layer

def get_fc_act(layer,idx):
    filter_size = layer.shape[1]
    act_layer = np.zeros((filter_size, 10000))
    for index in range(idx.shape[0]):
        act_layer[:,index+(i*(idx.shape[0]))] = layer[index].detach().numpy()
    return act_layer


for i, (data, target, idx) in enumerate(test_loader):

    Conv1out = F.relu(model.conv1(data))
    filter_sums_original_conv1 = get_conv_act(Conv1out,idx)
    Conv1out = F.max_pool2d(Conv1out, 2, 2)
    Conv2out = F.relu(model.conv2(Conv1out))
    filter_sums_original_conv2 = get_conv_act(Conv2out, idx)
    Conv2out = F.max_pool2d(Conv2out, 2, 2)
    Conv2Flatten = Conv2out.view(-1, 4 * 4 * 50)
    fc1out = F.relu(model.fc1(Conv2Flatten))
    filter_sums_original_fc1 = get_fc_act(fc1out,idx)
    fc2out = model.fc2(fc1out)
    logits = F.log_softmax(fc2out, dim=1)
    filter_sums_original_fc2 = get_fc_act(logits, idx)
    pred = logits.argmax(dim=1, keepdim=True)
    correct_original += pred.eq(target.view_as(pred)).sum().item()



    ############### Quantization Forward-pass #####################33


    x = quantize_tensor(data, bitwidth[0], min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

    x, scale_next, zero_point_next = quantizeLayer(x.tensor, q_model.conv1, stats['conv2'], x.scale,x.zero_point,bitwidth[1])
    QConv1out = dequantize_tensor(x, scale_next, zero_point_next)
    filter_sums_quantized_conv1 = get_conv_act(QConv1out,idx)
    x = F.max_pool2d(x, 2, 2)

    x, scale_next, zero_point_next = quantizeLayer(x, q_model.conv2, stats['fc1'], scale_next, zero_point_next, bitwidth[2])
    QConv2out = dequantize_tensor(x, scale_next, zero_point_next)
    filter_sums_quantized_conv2 = get_conv_act(QConv2out, idx)
    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 4 * 4 * 50)

    x, scale_next, zero_point_next = quantizeLayer(x, q_model.fc1, stats['fc2'], scale_next, zero_point_next, bitwidth[3])
    Qfc1out = dequantize_tensor(x, scale_next, zero_point_next)
    filter_sums_quantized_fc1 = get_fc_act(Qfc1out, idx)

    Qfc2out = q_model.fc2(Qfc1out)

    logitsQ= F.log_softmax(Qfc2out, dim=1)
    filter_sums_quantized_fc2 = get_fc_act(logitsQ, idx)

    pred = logitsQ.argmax(dim=1, keepdim=True)
    correct_quantized += pred.eq(target.view_as(pred)).sum().item()

print("Original  Model Accuracy: {:.4f}" .format(correct_original/len(dataset)))
print("Quantized Model Accuracy: {:.4f}" .format(correct_quantized/len(dataset)))

def KL (original , quantized):
    filter_sums_original =original
    filter_sums_quantized = quantized

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

    alpha = 1e-10 #KL stability
    KL = np.zeros((filter_sums_original.shape[0]))
    for it in range(filter_sums_original.shape[0]):
        KL[it] = math.log((std_quant[it]+alpha)/(std_original[it]+alpha))+(std_original[it]**2 - std_quant[it]**2 +(mean_original[it]-mean_quant[it])**2)/((2*std_quant[it]**2)+alpha)
    R= np.mean(KL)

    return R

R1 = KL(filter_sums_original_conv1,filter_sums_quantized_conv1)
R2 = KL(filter_sums_original_conv2,filter_sums_quantized_conv2)
R3 = KL(filter_sums_original_fc1, filter_sums_quantized_fc1)
R4 = KL(filter_sums_original_fc2, filter_sums_quantized_fc2)
print(R1)
print(R2)
print(R3)
print(R4)