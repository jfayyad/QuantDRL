"""## Rework Forward pass of Linear and Conv Layers to support Quantisation"""
import torch
from collections import namedtuple

"""## Rework Forward pass of Linear and Conv Layers to support Quantisation"""
import torch.nn.functional as F
from QuantFunc import calcScaleZeroPoint, calcScaleZeroPointSym, quantize_tensor, quantize_tensor_sym

def quantizeLayer(x, layer, stat, scale_x, zp_x, num_bits, vis=False, axs=None, X=None, y=None, sym=False):
    # for both conv and linear layers

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # WEIGHTS SIMULATED QUANTISED

    # quantise weights, activations are already quantised
    if sym:
        w = quantize_tensor_sym(layer.weight.data, num_bits=num_bits[0])
        b = quantize_tensor_sym(layer.bias.data, num_bits=num_bits[1])
    else:
        w = quantize_tensor(layer.weight.data, num_bits=num_bits[0])  # Min and Max not required, not data dependent
        b = quantize_tensor(layer.bias.data, num_bits=num_bits[1])

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    ## END WEIGHTS QUANTISED SIMULATION

    # if vis:
        # axs[X, y].set_xlabel("Visualising weights of layer: ")
        # visualise(layer.weight.data, axs[X, y])

    # QUANTISED OP, USES SCALE AND ZERO POINT TO DO LAYER FORWARD PASS. (How does backprop change here ?)
    # This is Quantisation Arithmetic
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    if sym:
        scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['min'], max_val=stat['max'],num_bits=num_bits[2])
    else:
        scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'],num_bits=num_bits[2])

    # Preparing input by saturating range to num_bits range.
    if sym:
        X = x.float()
        layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data)
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data)
    else:
        X = x.float() - zp_x
        layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data - zp_w)
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

        WA = layer.weight.data
        BA = layer.bias.data

    # All int computation
    if sym:
        x = (layer(X))
    else:
        x = (layer(X)) + zero_point_next

        # cast to int
    x.round_()

    # Perform relu too
    x = F.relu(x)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next




def quantizeConv(x, layer, stat, scale_x, zp_x, num_bits, vis=False, axs=None, X=None, y=None, sym=False):
    # for both conv and linear layers



    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # WEIGHTS SIMULATED QUANTISED

    # quantise weights, activations are already quantised
    wc = []
    scale_wc = []
    zp_wc = []


    for i in range (layer.weight.data.shape[0]):
        if sym:
            temp = quantize_tensor_sym(layer.weight[i].data, num_bits=num_bits[0])
            wc.append(temp.tensor)
            scale_wc.append(temp.scale)
            zp_wc.append(temp.zero_point)

        else:
            temp = quantize_tensor(layer.weight[i].data, num_bits=num_bits[0])  # Min and Max not required, not data dependent
            wc.append(temp.tensor.float())
            scale_wc.append(temp.scale)
            zp_wc.append(temp.zero_point)

    w = torch.stack(wc)
    layer.weight.data = w.detach().clone()

    b = quantize_tensor(layer.bias.data, num_bits=num_bits[1])
    layer.bias.data = b.tensor.float()


    ## END WEIGHTS QUANTISED SIMULATION

    # QUANTISED OP, USES SCALE AND ZERO POINT TO DO LAYER FORWARD PASS. (How does backprop change here ?)
    # This is Quantisation Arithmetic
    # scale_w = w.scale
    # zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    if sym:
        scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['min'], max_val=stat['max'],num_bits=num_bits[2])
    else:
        scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'],num_bits=num_bits[2])

    # Preparing input by saturating range to num_bits range.
    if sym:
        X = x.float()
        for i in range(layer.weight.data.shape[0]):
            # layer.weight[i].data = ((scale_x * scale_wc[i]) / scale_next) * (layer.weight[i].data)
            w[i] = ((scale_x * scale_wc[i]) / scale_next) * (layer.weight[i].data)

        layer.bias.data = (scale_b / scale_next) * (layer.bias.data)
    else:
        X = x.float() - zp_x
        for i in range(layer.weight.data.shape[0]):
            # layer.weight[i].data = ((scale_x * scale_wc[i]) / scale_next) * (layer.weight[i].data - zp_wc[i])
            w[i] = ((scale_x * scale_wc[i]) / scale_next) * (layer.weight[i].data - zp_wc[i])

        # temp = layer.weight.data

        layer.bias.data = (scale_b / scale_next) * (layer.bias.data - zp_b)

    layer.weight.data = w

    # All int computation
    if sym:
        x = (layer(X))
    else:
        x = (layer(X)) + zero_point_next

        # cast to int

    x.round_()

    # Perform relu too
    x = F.relu(x)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next

