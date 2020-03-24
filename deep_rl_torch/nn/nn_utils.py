import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


def act_funct_string2function(name):
    name = name.lower()
    if name == "relu":
        return F.relu
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "tanh":
        return torch.tanh


def query_act_funct(layer_dict):
    try:
        activation_function = act_funct_string2function(layer_dict["act_func"])
    except KeyError:
        def activation_function(x):
            return x
    return activation_function


def string2layer(name, input_size, neurons):
    name = name.lower()
    if name == "linear":
        return nn.Linear(input_size, neurons)
    elif name == "lstm":
        return nn.LSTM(input_size, neurons)
    elif name == "gru":
        return nn.GRU(input_size, neurons)


def create_ff_layers(input_size, layer_dict, output_size):
    layers = nn.ModuleList()
    act_functs = []
    for layer in layer_dict:
        this_layer_neurons = layer["neurons"]
        layers.append(string2layer(layer["name"], input_size, this_layer_neurons))
        act_functs.append(query_act_funct(layer))
        input_size = this_layer_neurons
    if output_size is not None:
        layers.append(nn.Linear(input_size, output_size))
        act_functs.append(None)
    return layers, act_functs


# Create a module list of conv layers specified in layer_dict
def create_conv_layers(input_matrix_shape, layer_dict):
    # format for entry in matrix_layers: ("conv", channels_in, channels_out, kernel_size, stride) if conv or
    #  ("batchnorm") for batchnorm

    channel_last_layer = input_matrix_shape[0]
    matrix_width = input_matrix_shape[1]
    matrix_height = input_matrix_shape[2]

    act_functs = []
    layers = nn.ModuleList()
    for layer in layer_dict:
        # Layer:
        if layer["name"] == "batchnorm":
            layers.append(nn.BatchNorm2d(channel_last_layer))
        elif layer["name"] == "conv":
            this_layer_channels = layer["filters"]
            layers.append(nn.Conv2d(channel_last_layer, this_layer_channels, layer["kernel_size"],
                                    layer["stride"]))
            matrix_width = conv2d_size_out(matrix_width, layer["kernel_size"], layer["stride"])
            matrix_height = conv2d_size_out(matrix_height, layer["kernel_size"], layer["stride"])
            channel_last_layer = this_layer_channels

        act_functs.append(query_act_funct(layer))

    conv_output_size = matrix_width * matrix_height * channel_last_layer
    return layers, conv_output_size, act_functs


def apply_layers(x, layers, act_functs):
    for idx in range(len(layers)):
        layer = layers[idx]
        act_func = act_functs[idx]
        x = layer(x)
        if act_functs[idx] is not None:
            x = act_func(x)
    return x


def one_hot_encode(x, num_actions):
    y = torch.zeros(x.shape[0], num_actions).float()
    return y.scatter(1, x, 1)



def calc_gradient_norm(layers):
    grads = [p.grad.data for p in layers.parameters()]
    return calc_list_norm(grads)
    #total_norm = 0
    #for p in layers.parameters():
    #    param_norm = p.grad.data.norm(2)
    #    total_norm += param_norm.item() ** 2
    #return total_norm ** (1. / 2)

def calc_norm(layers):
    params = layers.parameters()
    return calc_list_norm(params)

    # total_norm = torch.tensor(0.)
    # for param in layers.parameters():
    #     total_norm += torch.norm(param)
    # return total_norm

def calc_list_norm(layer_list):
    total_norm = torch.tensor(0.)
    for param in layer_list:
        total_norm += torch.norm(param)
    return total_norm.item()

def calc_list_norm_std(layer_list):
    all_norms = []
    total_norm = torch.tensor(0.)
    for param in layer_list:
        norm = torch.norm(param)

        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        total_norm += norm
        all_norms.append(norm)
    return torch.sqrt(total_norm), torch.std(torch.tensor(all_norms))

def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.get_updateable_params(), net.get_updateable_params()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def hard_update(net, net_target):
    for param_target, param in zip(net_target.get_updateable_params(), net.get_updateable_params()):
        param_target.data.copy_(param.data)
