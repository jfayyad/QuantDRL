import torch
import torch.nn as nn
import torch.nn.functional as F
from QuantLayerRes import quantizeConvBnRelu, quantizeConvBn, quantizeFc
from QuantFunc import quantize_tensor, dequantize_tensor
import numpy as np
from Dataset import MyDataset



def residu(L_out_xq, L_residualq):
    dq_out_xq = dequantize_tensor(L_out_xq[0], L_out_xq[1], L_out_xq[2])
    dq_residualq = dequantize_tensor(L_residualq[0], L_residualq[1], L_residualq[2])
    dq_out_xq += dq_residualq
    dq_out_xq = F.relu(dq_out_xq)
    out_xq = quantize_tensor(dq_out_xq, 8)
    return out_xq.tensor, out_xq.scale, out_xq.zero_point

def get_conv_act(layer):
    filter_size = layer.shape[1]
    act_layer = np.zeros((filter_size, bs))
    layer = layer.reshape(layer.detach().numpy().shape[0], layer.detach().numpy().shape[1], -1)
    for filter_it in range(layer.shape[1]):
        layer_filter = layer[:, filter_it, :]
        act_layer[filter_it, :] = np.mean(layer_filter.detach().numpy(), axis=1)
    return act_layer

def get_fc_act(layer):
    filter_size = layer.shape[1]
    act_layer = np.zeros((filter_size, bs))
    for index in range(layer.shape[0]):
        act_layer[:, index] = layer[index].detach().numpy()
    return act_layer

# global R




def KLD(x,xq,sc,zp):


    act = get_conv_act(x)
    dq_xq = dequantize_tensor(xq, sc, zp)
    Quant_act = get_conv_act(dq_xq)

    mean_original = np.mean(act, axis=1)
    std_original = np.std(act, axis=1)

    mean_quant = np.mean(Quant_act, axis=1)
    std_quant = np.std(Quant_act, axis=1)

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



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu2 = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, z):
        x, xq, scale_next, zero_point_next, bits = z[0], z[1], z[2], z[3], z[4]

        residual = x
        residualq = xq

        out = self.conv1(x)
        out_xq, out_scale_next, out_zero_point_next = quantizeConvBnRelu(x, xq, self.conv1, scale_next, zero_point_next, bits[0], bn_fake=Quant_BN)
        # dq_xq = dequantize_tensor(out_xq.clone().detach(), out_scale_next, out_zero_point_next)

        out = self.conv2(out)
        out_xq, out_scale_next, out_zero_point_next = quantizeConvBn(out, out_xq, self.conv2, out_scale_next, out_zero_point_next, bits[1], bn_fake=Quant_BN)
        # dq_xq = dequantize_tensor(out_xq.clone().detach(), out_scale_next, out_zero_point_next)

        if self.downsample:
            residual = self.downsample(x)
            residualq, scale_next, zero_point_next = quantizeConvBn(x, residualq, self.downsample, scale_next, zero_point_next, bits[2], bn_fake=Quant_BN)

        out += residual
        L_out_xq = [out_xq, out_scale_next, out_zero_point_next]
        L_residualq = [residualq, scale_next, zero_point_next]
        out_xq, out_scale_next, out_zero_point_next = residu(L_out_xq, L_residualq)

        out = self.relu2(out)
        # out_xq = F.relu(out_xq)

        return out, out_xq, out_scale_next, out_zero_point_next, bits


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        Rkld = []
        ################################### CONV1 and CONV1 Quant ####################################
        xq = quantize_tensor(x, bitwidth[0])
        x = self.conv1(x)
        x = self.maxpool(x)
        xq, scale_next, zero_point_next = quantizeConvBnRelu(x, xq.tensor, self.conv1, xq.scale, xq.zero_point, bitwidth[1], bn_fake=Quant_BN)
        xq = self.maxpool(xq)


        Rkld.append(KLD(x,xq,scale_next,zero_point_next))


        ################################### Blocks and Blocks Quant ###################################
        x, xq, scale_next, zero_point_next, _ = self.layer0([x, xq, scale_next, zero_point_next, bitwidth[2]])
        # dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)
        Rkld.append(KLD(x,xq,scale_next,zero_point_next))

        x, xq, scale_next, zero_point_next, _ = self.layer1([x, xq, scale_next, zero_point_next, bitwidth[3]])
        dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)
        Rkld.append(KLD(x,xq,scale_next,zero_point_next))

        x, xq, scale_next, zero_point_next, _ = self.layer2([x, xq, scale_next, zero_point_next, bitwidth[4]])
        # dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)
        Rkld.append(KLD(x, xq, scale_next, zero_point_next))

        x, xq, scale_next, zero_point_next, _ = self.layer3([x, xq, scale_next, zero_point_next, bitwidth[5]])
        # dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)
        Rkld.append(KLD(x,xq,scale_next,zero_point_next))

        ################################### FCs and FCs Quant ###################################
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        xq = self.avgpool(xq.float())
        xq = xq.view(xq.size(0), -1)

        x = self.fc(x)
        xq, scale_next, zero_point_next = quantizeFc(x, xq, self.fc, scale_next, zero_point_next, bitwidth[6])
        xq = dequantize_tensor(xq, scale_next, zero_point_next)

        return x, xq, Rkld


########## bitwidth #########

Quant_BN = False
bitwidth = [8,                                                                              # bitwidth[0] = input (act)
            [[8, 8], 8, [8, 8], 8],                                                         # bitwidth[1] = conv1_bn1: [ conv1[w, b], act, bn[w, b], act]
            [[[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8]],     # bitwidth[2] = block0{conv1_conv2_downsample}
            [[[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8]],     # bitwidth[3] = block1{conv1_conv2_downsample}
            [[[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8]],     # bitwidth[4] = block2{conv1_conv2_downsample}
            [[[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8] , [[8, 8], 8, [8, 8], 8]],     # bitwidth[5] = block3{conv1_conv2_downsample}
            [[8, 8], 8]                                                                     # bitwidth[6] = fc:[[w, b], act]
            ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(ResidualBlock, [2, 2, 2, 2])
model.load_state_dict(torch.load("Res18.pt", map_location="cpu"))
model = model.to(device)
model.eval()

bs = 100
n = 3
l = bs*n


dataset = MyDataset()
kwargs = {'num_workers': 0, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(dataset,batch_size=bs, shuffle=False, **kwargs)
correct = 0
correct_quant = 0
# Rconv = np.array([])
# Rlayer0 = np.array([])
# Rlayer1 = np.array([])
# Rlayer2 = np.array([])
# Rlayer3 = np.array([])
R = []

for i, (data, target, idx) in enumerate(test_loader):
    data,target = data.to(device), target.to(device)
    out, out_quant, R_kld = model(data)
    pred_orig = out.argmax(dim=1, keepdim=True)
    pred_quant = out_quant.argmax(dim=1, keepdim=True)
    correct += pred_orig.eq(target.view_as(pred_orig)).sum().item()
    correct_quant += pred_quant.eq(target.view_as(pred_quant)).sum().item()
    R.append(R_kld)


    if i == n-1:
      break

Rmean = np.mean(R, axis=0)
print('\n Original Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, l,
        100. * correct / l))

print('\n Quantization Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct_quant, l,
        100. * correct_quant / l))








