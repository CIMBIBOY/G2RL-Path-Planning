import torch
import torch.nn as nn
from functools import reduce

def print_model_summary(model, input_size, batch_size):
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

    model.apply(register_hook)
    # Move the input tensor to the same device as the model
    model(torch.zeros(*input_size).to(device))

    for h in hooks:
        h.remove()

    print('-------------------------------------------------------')
    print(f'{"Layer (type)":<25} {"Output Shape":<20} {"Param #":<10}')
    print('=======================================================')
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = f'{layer:<25} {str(summary[layer]["output_shape"]):<20} {summary[layer]["nb_params"]:<10}'
        total_params += summary[layer]['nb_params']
        if summary[layer]['output_shape'] != 'multiple_outputs':
            product = 1
            for dim in summary[layer]['output_shape']:
                if isinstance(dim, int):
                    product *= dim
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

