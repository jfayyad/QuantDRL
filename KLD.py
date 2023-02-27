import torch
from Model import Net
import copy
from torchvision import datasets, transforms
import torch.nn.functional as F
from QuantFunc import quantize_tensor, dequantize_tensor
from Stats import gatherStats
from QuantLayer import quantizeConv, quantizeLayer
import numpy as np
import math
from Model import batch_s
import time

######## Models #######
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()
q_model = copy.deepcopy(model)

######## DataLoaders #######
kwargs = {'num_workers': 4, 'pin_memory': False}


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

test_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_s, shuffle=False, **kwargs)

###### Quantized Model #######

def get_conv_act(layer):
    filter_size = layer.shape[1]
    act_layer = np.zeros((filter_size, batch_s))
    layer = layer.reshape(layer.detach().numpy().shape[0], layer.detach().numpy().shape[1], -1)
    for filter_it in range(layer.shape[1]):
        layer_filter = layer[:, filter_it, :]
        act_layer[filter_it, :] = np.mean(layer_filter.detach().numpy(), axis=1)
    return act_layer


def get_fc_act(layer):
    filter_size = layer.shape[1]
    act_layer = np.zeros((filter_size, batch_s))
    for index in range(layer.shape[0]):
        act_layer[:, index] = layer[index].detach().numpy()
    return act_layer


def KL (original, quantized):
    mean_original = np.mean(original, axis=1)
    std_original = np.std(original, axis=1)

    mean_quant = np.mean(quantized, axis=1)
    std_quant = np.std(quantized, axis=1)

    alpha = 1e-10 #KL stability
    KL = np.log10(((std_quant + alpha) / (std_original + alpha)) + (std_original ** 2 - std_quant ** 2 +
                    (mean_original - mean_quant) ** 2) / ((2 * std_quant ** 2) + alpha))
    R = np.mean(KL)

    # # calculate the outliers for kernel-wise quantization:
    # Q1 = np.quantile(KL, .25)
    # Q3 = np.quantile(KL, .75)
    # IQR = Q3 - Q1
    # UL = Q3 + 1.5*IQR
    # LL = 1e-4 #LL = Q1 - 1.5*IQR
    # idx_uv = [i for i,v in enumerate(KL >= UL) if v]
    # idx_lv = [i for i, v in enumerate(KL <= LL) if v]

    return R


def acc_kl(model, q_model, bitwidth):
    correct_original = 0
    correct_quantized = 0
    R1 = np.array([])
    R2 = np.array([])
    R3 = np.array([])
    R4 = np.array([])
    # stats = gatherStats(q_model, test_loader)
    for i, (data, target, idx) in enumerate(test_loader):
        Conv1out = F.relu(model.conv1(data))
        conv1_act = get_conv_act(Conv1out)
        Conv1out = F.max_pool2d(Conv1out, 2, 2)
        Conv2out = F.relu(model.conv2(Conv1out))
        conv2_act = get_conv_act(Conv2out)
        Conv2out = F.max_pool2d(Conv2out, 2, 2)
        Conv2Flatten = Conv2out.view(-1, 4 * 4 * 50)
        fc1out = F.relu(model.fc1(Conv2Flatten))
        fc1_act = get_fc_act(fc1out)
        fc2out = model.fc2(fc1out)
        logits = F.log_softmax(fc2out, dim=1)
        fc2_act = get_fc_act(logits)
        pred = logits.argmax(dim=1, keepdim=True)
        correct_original += pred.eq(target.view_as(pred)).sum().item()

        ############### Quantization Forward-pass #####################33

        x = quantize_tensor(data, bitwidth[0], min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

        x, scale_next, zero_point_next = quantizeLayer(x.tensor, q_model.conv1, stats['conv2'], x.scale,x.zero_point,bitwidth[1])
        QConv1out = dequantize_tensor(x.clone().detach(), scale_next, zero_point_next)
        Qconv1_act = get_conv_act(QConv1out)
        x = F.max_pool2d(x, 2, 2)

        x, scale_next, zero_point_next = quantizeLayer(x, q_model.conv2, stats['fc1'], scale_next, zero_point_next, bitwidth[2])
        QConv2out = dequantize_tensor(x.clone().detach(), scale_next, zero_point_next)
        Qconv2_act = get_conv_act(QConv2out)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4 * 4 * 50)

        x, scale_next, zero_point_next = quantizeLayer(x, q_model.fc1, stats['fc2'], scale_next, zero_point_next, bitwidth[3])
        Qfc1out = dequantize_tensor(x, scale_next, zero_point_next)
        Qfc1_act = get_fc_act(Qfc1out)

        Qfc2out = q_model.fc2(Qfc1out)

        logitsQ= F.log_softmax(Qfc2out, dim=1)
        Qfc2_act = get_fc_act(logitsQ)

        predQ = logitsQ.argmax(dim=1, keepdim=True)
        correct_quantized += predQ.eq(target.view_as(predQ)).sum().item()

        R1 = np.append(R1, KL(conv1_act, Qconv1_act))
        R2 = np.append(R2, KL(conv2_act, Qconv2_act))
        R3 = np.append(R3, KL(fc1_act, Qfc1_act))
        R4 = np.append(R4, KL(fc2_act, Qfc2_act))

    acc = [correct_original/len(dataset), correct_quantized/len(dataset)]
    R = [np.mean(R1), np.mean(R2), np.mean(R3), np.mean(R4)]
    print("  Original  Model Accuracy: {:.4f}".format(acc[0]))
    print("  Quantized Model Accuracy: {:.4f}".format(acc[1]))
    print("  KL-D Values = [{}, {}, {}, {}]".format(R[0], R[1], R[2], R[3]))
    return acc, R


def acc_org(model):
    correct_original = 0
    for i, (data, target, idx) in enumerate(test_loader):
        Conv1out = F.relu(model.conv1(data))
        Conv1out = F.max_pool2d(Conv1out, 2, 2)
        Conv2out = F.relu(model.conv2(Conv1out))
        Conv2out = F.max_pool2d(Conv2out, 2, 2)
        Conv2Flatten = Conv2out.view(-1, 4 * 4 * 50)
        fc1out = F.relu(model.fc1(Conv2Flatten))
        fc2out = model.fc2(fc1out)
        logits = F.log_softmax(fc2out, dim=1)
        pred = logits.argmax(dim=1, keepdim=True)
        correct_original += pred.eq(target.view_as(pred)).sum().item()

    acc = correct_original/len(dataset)
    return acc


def acc_quant(q_model, bitwidth):
    correct_quantized = 0
    for i, (data, target, idx) in enumerate(test_loader):
        x = quantize_tensor(data, bitwidth[0], min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])
        x, scale_next, zero_point_next = quantizeLayer(x.tensor, q_model.conv1, stats['conv2'], x.scale, x.zero_point, bitwidth[1])
        x = F.max_pool2d(x, 2, 2)
        x, scale_next, zero_point_next = quantizeLayer(x, q_model.conv2, stats['fc1'], scale_next, zero_point_next, bitwidth[2])
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x, scale_next, zero_point_next = quantizeLayer(x, q_model.fc1, stats['fc2'], scale_next, zero_point_next, bitwidth[3])
        Qfc1out = dequantize_tensor(x, scale_next, zero_point_next)
        Qfc2out = q_model.fc2(Qfc1out)
        logitsQ= F.log_softmax(Qfc2out, dim=1)
        predQ = logitsQ.argmax(dim=1, keepdim=True)
        correct_quantized += predQ.eq(target.view_as(predQ)).sum().item()

    acc = correct_quantized/len(dataset)
    print("  Quantized Model Accuracy: {:.4f}".format(acc))
    return acc

stats = gatherStats(model, test_loader)


bitwidth = [8, [8, 8, 8], [8, 8, 8], [8, 8, 8]]

t = time.time()
acc_quant(q_model, bitwidth)
# accu, R = acc_kl(model, q_model, bitwidth)
# print('R_total = ', - np.log10(np.sum(R)))
elapsed = time.time() - t
print(elapsed)