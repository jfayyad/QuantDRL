from QuantFunc import calcScaleZeroPoint, calcScaleZeroPointSym, quantize_tensor, quantize_tensor_sym, dequantize_tensor
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# from channelwise_quant import *


def quantizeConvBnRelu(x, xq, block, scale_x, zp_x, num_bits):
    # for both conv and bn layers (relu included too)

    layer = block[0]
    bn = block[1]

    # cache old values
    W = layer.weight.data.clone()
    if layer.bias is not None:
        B = layer.bias.data.clone()

    Wbn = bn.weight.data
    Bbn = bn.bias.data

    ################################ Conv2d WEIGHTS SIMULATED QUANTISED ################################
    # quantise weights, activations are already quantised
    tensor_list=[]
    scale_list =[]
    zp_list = []

    if isinstance(num_bits[0], list):
        cw_bits = num_bits[0]
    else:
        cw_bits = [num_bits[0]] * (W.shape[0])

    for i in range(W.shape[0]):
        w = quantize_tensor(layer.weight.data[i], num_bits=cw_bits[i])  # Min and Max not required, not data dependent
        tensor_list.append(w.tensor.float())
        scale_list.append(w.scale)
        zp_list.append(w.zero_point)
    if layer.bias is not None:
        b = quantize_tensor(layer.bias.data, num_bits=num_bits[0])

        layer.bias.data = b.tensor.float()

        scale_b = b.scale
        zp_b = b.zero_point

    min_val, max_val = x.min(), x.max()
    scale_next, zero_point_next = calcScaleZeroPoint(min_val= min_val, max_val=max_val,num_bits=num_bits[1])

    # Preparing input by saturating range to num_bits range.
    X = xq.float() - zp_x

    for i in range(W.shape[0]):
        layer.weight.data[i] = ((scale_x * scale_list[i]) / scale_next) * (tensor_list[i] - zp_list[i])

    if layer.bias is not None:
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

    # All int computation
    xq = (layer(X)) + zero_point_next

    # cast to int and dequantize
    xq.round_()

    # Bypass BatchNorm quantization
    dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)
    xq = bn(dq_xq)

    xq = F.relu(xq)
    xq = quantize_tensor(xq, num_bits=8)

    ############################################### RelU #################################################
    # xq = F.relu(xq)

    # Reset weights for next forward pass
    layer.weight.data = W
    if layer.bias is not None:
        layer.bias.data = B

    bn.weight.data = Wbn
    bn.bias.data = Bbn

    # return xq, scale_next_bn, zero_point_next_bn
    return xq.tensor.float(), xq.scale, xq.zero_point


def quantizeConvBn(x, xq, block, scale_x, zp_x, num_bits):
    # for both conv and bn layers

    layer = block[0]
    bn = block[1]

    # cache old values
    W = layer.weight.data.clone()
    if layer.bias is not None:
        B = layer.bias.data.clone()

    Wbn = bn.weight.data
    Bbn = bn.bias.data

    ################################ Conv2d WEIGHTS SIMULATED QUANTISED ################################
    # quantise weights, activations are already quantised
    tensor_list = []
    scale_list = []
    zp_list = []
    if isinstance(num_bits[0], list):
        cw_bits = num_bits[0]
    else:
        cw_bits = [num_bits[0]] * (W.shape[0])

    for i in range(W.shape[0]):
        w = quantize_tensor(layer.weight.data[i], num_bits=cw_bits[i])  # Min and Max not required, not data dependent
        tensor_list.append(w.tensor.float())
        scale_list.append(w.scale)
        zp_list.append(w.zero_point)
    if layer.bias is not None:
        b = quantize_tensor(layer.bias.data, num_bits=num_bits[0])

        layer.bias.data = b.tensor.float()

        scale_b = b.scale
        zp_b = b.zero_point

    min_val, max_val = x.min(), x.max()
    scale_next, zero_point_next = calcScaleZeroPoint(min_val=min_val, max_val=max_val, num_bits=num_bits[1])

    # Preparing input by saturating range to num_bits range.
    X = xq.float() - zp_x

    for i in range(W.shape[0]):
        layer.weight.data[i] = ((scale_x * scale_list[i]) / scale_next) * (tensor_list[i] - zp_list[i])
    if layer.bias is not None:
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

    # All int computation
    xq = (layer(X)) + zero_point_next

    # cast to int and dequantize
    xq.round_()

    # ################################ batchnorm2d WEIGHT/BIAS SIMULATED QUANTISED ################################
    # Bypass BatchNorm quantization
    dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)
    xq = bn(dq_xq)

    xq = quantize_tensor(xq, num_bits=8)

    ############################################### RelU #################################################
    # xq = F.relu(xq)

    # Reset weights for next forward pass
    layer.weight.data = W
    if layer.bias is not None:
        layer.bias.data = B

    bn.weight.data = Wbn
    bn.bias.data = Bbn

    # return xq, scale_next_bn, zero_point_next_bn
    return xq.tensor.float(), xq.scale, xq.zero_point


def quantizeFc(x, xq, layer, scale_x, zp_x, num_bits):
    # for linear layers
    # cache old values
    W = layer.weight.data
    if layer.bias is not None:
        B = layer.bias.data

    ################################ Conv2d WEIGHTS SIMULATED QUANTISED ################################
    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data, num_bits=num_bits[0])  # Min and Max not required, not data dependent
    if layer.bias is not None:
        b = quantize_tensor(layer.bias.data, num_bits=num_bits[0])

    layer.weight.data = w.tensor.float()
    if layer.bias is not None:
        layer.bias.data = b.tensor.float()

    scale_w = w.scale
    zp_w = w.zero_point
    if layer.bias is not None:
        scale_b = b.scale
        zp_b = b.zero_point

    min_val, max_val = x.min(), x.max()
    scale_next, zero_point_next = calcScaleZeroPoint(min_val= min_val, max_val=max_val,num_bits=num_bits[1])

    # Preparing input by saturating range to num_bits range.
    X = xq.float() - zp_x
    layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data - zp_w)
    if layer.bias is not None:
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

    # All int computation
    xq = (layer(X)) + zero_point_next

    # cast to int and dequantize
    xq.round_()

    xq = F.relu(xq)

    # Reset weights for next forward pass
    layer.weight.data = W
    if layer.bias is not None:
        layer.bias.data = B

    return xq, scale_next, zero_point_next