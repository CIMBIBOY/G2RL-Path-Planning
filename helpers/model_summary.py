import numpy as np
from collections import OrderedDict
from functools import reduce
import torch
import torch.nn as nn

def print_model_summary_dqn(model, input_size, batch_size):
    def register_hook(module):
        def hook(module, input, output):
            nonlocal module_idx
            class_name = module.__class__.__name__
            module_idx += 1

            m_key = f'{class_name}-{module_idx}'
            summary[m_key] = {}
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [
                    [-1] + list(o.size())[1:] if isinstance(o, torch.Tensor) else 'multiple_outputs' for o in output
                ]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = batch_size

            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += reduce(lambda x, y: x * y, module.weight.size())
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += reduce(lambda x, y: x * y, module.bias.size())
            summary[m_key]['nb_params'] = params

        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and module != model:
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    module_idx = 0

    # Get the device of the model
    device = next(model.parameters()).device

    done = torch.zeros(input_size[0])
    lstm_state = (
        torch.zeros(model.lstm.num_layers, input_size[0], model.lstm.hidden_size).to(device),
        torch.zeros(model.lstm.num_layers, input_size[0], model.lstm.hidden_size).to(device)
    )

    model.apply(register_hook)
    # Move the input tensor to the same device as the model
    model(torch.zeros(*input_size).to(device), lstm_state, done)

    for h in hooks:
        h.remove()

    print('-------------------------------------------------------')
    print(f'Input Shape: {input_size}')
    print('-------------------------------------------------------')
    print(f'{"Layer (type)":<25} {"Output Shape":<20} {"Param #":<10}')
    print('=======================================================')
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # Convert output shape to string, handling 'multiple_outputs' case
        if isinstance(summary[layer]["output_shape"], list) and summary[layer]["output_shape"][0] == 'multiple_outputs':
            output_shape_str = str(summary[layer]["output_shape"])
        else:
            output_shape_str = str(summary[layer]["output_shape"])
        
        line_new = f'{layer:<25} {output_shape_str:<20} {summary[layer]["nb_params"]:<10}'
        total_params += summary[layer]['nb_params']
        if summary[layer]['output_shape'] != 'multiple_outputs':
            if isinstance(summary[layer]['output_shape'], list):
                int_dims = [dim for dim in summary[layer]['output_shape'] if isinstance(dim, int)]
                if int_dims:
                    product = reduce(lambda x, y: x * y, int_dims)
                    total_output += product
            else:
                int_dims = [dim for dim in summary[layer]['output_shape'] if isinstance(dim, int)]
                if int_dims:
                    product = reduce(lambda x, y: x * y, int_dims)
                    total_output += product
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable']:
                trainable_params += summary[layer]['nb_params']
        print(line_new)

    print('=======================================================')
    print(f'Total params: {total_params}')
    print(f'Trainable params: {trainable_params}')
    print(f'Non-trainable params: {total_params - trainable_params}')
    print('-------------------------------------------------------')

def print_model_summary_ppo(model, input_size, device):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())

            # Preserve the batch size from input throughout the network
            batch_size = input[0].size(0)
            summary[m_key]["input_shape"][0] = batch_size

            if isinstance(output, (list, tuple)):
                # If the output is a tuple, process each element
                summary[m_key]["output_shape"] = [
                    [batch_size] + list(o.size())[1:] if isinstance(o, torch.Tensor) else 'multiple_outputs' for o in output
                ]
            else:
                summary[m_key]["output_shape"] = [batch_size] + list(output.size())[1:]

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.tensor(list(module.weight.size()))).item()
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.tensor(list(module.bias.size()))).item()
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # Create the summary dictionary and list for hooks
    summary = OrderedDict()
    hooks = []

    # Register the hooks
    model.apply(register_hook)

    # Make a forward pass with dummy data
    x = torch.zeros(input_size).to(device)
    lstm_state = (
        torch.zeros(model.lstm.num_layers, input_size[1], model.lstm.hidden_size).to(device),
        torch.zeros(model.lstm.num_layers, input_size[1], model.lstm.hidden_size).to(device)
    )
    done = torch.zeros(input_size[1]).to(device)
    mask = torch.zeros((input_size[0], input_size[1], 5)).to(device)
    
    with torch.no_grad():
        model.get_action_and_value(x, lstm_state, done, mask)

    # Remove the hooks
    for h in hooks:
        h.remove()

    # Print the correct input shape at the top
    print("----------------------------------------------------------------")
    print(f'------------- Input shape: {input_size} -------------')
    print("batch_size, num_envs, time_dim, obs_width, obs_height, channels)")
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]['nb_params']
        if isinstance(summary[layer]["output_shape"], list):
            for shape in summary[layer]["output_shape"]:
                if isinstance(shape, list):
                    total_output += np.prod(shape)
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # Assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")