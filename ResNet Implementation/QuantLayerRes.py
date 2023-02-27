from QuantFunc import calcScaleZeroPoint, calcScaleZeroPointSym, quantize_tensor, quantize_tensor_sym, dequantize_tensor
import torch.nn as nn
import torch.nn.functional as F


def fakeQuant(layer, num_bits):

    w = quantize_tensor(layer.weight.data, num_bits=num_bits[0])  # Min and Max not required, not data dependent
    b = quantize_tensor(layer.bias.data, num_bits=num_bits[1])

    dq_w = dequantize_tensor(w.tensor, w.scale, w.zero_point)
    dq_b = dequantize_tensor(b.tensor, b.scale, b.zero_point)

    layer.weight.data = dq_w
    layer.bias.data = dq_b

    return layer


def quantizeConvBnRelu(x, xq, block, scale_x, zp_x, num_bits, bn_fake=False):
    # for both conv and bn layers (relu included too)

    layer = block[0]
    bn = block[1]

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    Wbn = bn.weight.data
    Bbn = bn.bias.data

    ################################ Conv2d WEIGHTS SIMULATED QUANTISED ################################
    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data, num_bits=num_bits[0][0])  # Min and Max not required, not data dependent
    b = quantize_tensor(layer.bias.data, num_bits=num_bits[0][1])

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    min_val, max_val = x.min(), x.max()
    scale_next, zero_point_next = calcScaleZeroPoint(min_val= min_val, max_val=max_val,num_bits=num_bits[1])

    # Preparing input by saturating range to num_bits range.
    X = xq.float() - zp_x
    layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data - zp_w)
    layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

    # All int computation
    xq = (layer(X)) + zero_point_next

    # cast to int and dequantize
    xq.round_()

    # ################################ batchnorm2d WEIGHT/BIAS SIMULATED QUANTISED ################################
    # # quantise weights, activations are already quantised
    # w_bn = quantize_tensor(bn.weight.data, num_bits=num_bits[0])  # Min and Max not required, not data dependent
    # b_bn = quantize_tensor(bn.bias.data, num_bits=num_bits[1])
    #
    # bn.weight.data = w_bn.tensor.float()
    # bn.bias.data = b_bn.tensor.float()
    #
    # scale_w_bn = w_bn.scale
    # zp_w_bn = w_bn.zero_point
    # scale_b_bn = b_bn.scale
    # zp_b_bn = b_bn.zero_point
    #
    # min_val_bn, max_val_bn = xconv.min(), xconv.max()
    # scale_next_bn, zero_point_next_bn = calcScaleZeroPoint(min_val= min_val_bn, max_val=max_val_bn, num_bits=num_bits[2])
    #
    # # Preparing input by saturating range to num_bits range.
    # X_bn = xq.float() - zero_point_next
    # bn.weight.data = ((scale_next * scale_w_bn) / scale_next_bn) * (bn.weight.data - zp_w_bn)
    # bn.bias.data = (scale_b_bn / scale_next_bn) * (bn.bias.data - zp_b_bn)
    # # All int computation
    # xq = bn(X_bn) + zero_point_next_bn
    #
    # # cast to int and dequantize
    # xq.round_()

    # Bypass BatchNorm quantization
    dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)

    if bn_fake == True:
        bn_fake = fakeQuant(layer=bn, num_bits=num_bits[2])
        xq = bn_fake(dq_xq)
    else:
        xq = bn(dq_xq)

    xq = F.relu(xq)
    xq = quantize_tensor(xq, num_bits[3])

    ############################################### RelU #################################################
    # xq = F.relu(xq)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    bn.weight.data = Wbn
    bn.bias.data = Bbn

    # return xq, scale_next_bn, zero_point_next_bn
    return xq.tensor.float(), xq.scale, xq.zero_point


def quantizeConvBn(x, xq, block, scale_x, zp_x, num_bits, bn_fake=False):
    # for both conv and bn layers

    layer = block[0]
    bn = block[1]
    # xconv = layer(x)  # to find min and max values before BN

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    Wbn = bn.weight.data
    Bbn = bn.bias.data

    ################################ Conv2d WEIGHTS SIMULATED QUANTISED ################################
    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data, num_bits=num_bits[0][0])  # Min and Max not required, not data dependent
    b = quantize_tensor(layer.bias.data, num_bits=num_bits[0][1])

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    min_val, max_val = x.min(), x.max()
    scale_next, zero_point_next = calcScaleZeroPoint(min_val= min_val, max_val=max_val,num_bits=num_bits[1])

    # Preparing input by saturating range to num_bits range.
    X = xq.float() - zp_x
    layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data - zp_w)
    layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

    # All int computation
    xq = (layer(X)) + zero_point_next

    # cast to int and dequantize
    xq.round_()

    # ################################ batchnorm2d WEIGHT/BIAS SIMULATED QUANTISED ################################
    # Bypass BatchNorm quantization
    dq_xq = dequantize_tensor(xq.clone().detach(), scale_next, zero_point_next)

    if bn_fake == True:
        bn_fake = fakeQuant(layer=bn, num_bits=num_bits[2])
        xq = bn_fake(dq_xq)
    else:
        xq = bn(dq_xq)

    xq = quantize_tensor(xq, num_bits[3])

    ############################################### RelU #################################################
    # xq = F.relu(xq)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    bn.weight.data = Wbn
    bn.bias.data = Bbn

    # return xq, scale_next_bn, zero_point_next_bn
    return xq.tensor.float(), xq.scale, xq.zero_point


def quantizeFc(x, xq, layer, scale_x, zp_x, num_bits):
    # for linear layers
    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    ################################ Conv2d WEIGHTS SIMULATED QUANTISED ################################
    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data, num_bits=num_bits[0][0])  # Min and Max not required, not data dependent
    b = quantize_tensor(layer.bias.data, num_bits=num_bits[0][1])

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    min_val, max_val = x.min(), x.max()
    scale_next, zero_point_next = calcScaleZeroPoint(min_val= min_val, max_val=max_val,num_bits=num_bits[1])

    # Preparing input by saturating range to num_bits range.
    X = xq.float() - zp_x
    layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data - zp_w)
    layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

    # All int computation
    xq = (layer(X)) + zero_point_next

    # cast to int and dequantize
    xq.round_()

    xq = F.relu(xq)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    return xq, scale_next, zero_point_next