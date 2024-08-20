import torch
import torch.nn as nn
from functools import reduce
from collections import OrderedDict

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

    model.apply(register_hook)
    # Move the input tensor to the same device as the model
    model(torch.zeros(*input_size).to(device))

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

def print_model_summary_ppo(model, input_size, batch_size, env, device):
    def register_hook(module):
        def hook(module, input, output):
            nonlocal module_idx
            class_name = module.__class__.__name__
            module_idx += 1

            m_key = f'{class_name}-{module_idx}'
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = batch_size

            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [
                    list(o.size()) if isinstance(o, torch.Tensor) else 'multiple_outputs' for o in output
                ]
                summary[m_key]['output_shape'] = [
                    [-1] + o[1:] if isinstance(o, list) else o for o in summary[m_key]['output_shape']
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

    summary = OrderedDict()
    hooks = []
    module_idx = 0

    model.apply(register_hook)

    # Move the input tensor to the same device as the model
    input_tensor = torch.zeros((1,) + input_size[1:]).to(device)
    
    # Adjust lstm_state to match the model's LSTM hidden size
    lstm_hidden_size = model.lstm.hidden_size
    lstm_num_layers = model.lstm.num_layers
    lstm_state = (
        torch.zeros(lstm_num_layers, 1, lstm_hidden_size).to(device),
        torch.zeros(lstm_num_layers, 1, lstm_hidden_size).to(device)
    )
    done = torch.zeros(1).to(device)
    
    # Perform a forward pass with dummy lstm_state and done
    with torch.no_grad():
        model.get_action_and_value(input_tensor, lstm_state, done, env, device)

    for h in hooks:
        h.remove()

    print('-------------------------------------------------------')
    print(f'Input Shape: {input_size}')
    print('-------------------------------------------------------')
    print(f'{"Layer (type)":<25} {"Output Shape":<30} {"Param #":<15}')
    print('=======================================================')
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # Convert output shape to string, handling 'multiple_outputs' case
        if isinstance(summary[layer]["output_shape"], list):
            output_shape_str = ', '.join([str(o) if isinstance(o, list) else str(o) for o in summary[layer]["output_shape"]])
        else:
            output_shape_str = str(summary[layer]["output_shape"])

        line_new = f'{layer:<25} {output_shape_str:<30} {summary[layer]["nb_params"]:<15}'
        total_params += summary[layer]['nb_params']

        if summary[layer]['output_shape'] != 'multiple_outputs':
            if isinstance(summary[layer]['output_shape'], list):
                for shape in summary[layer]['output_shape']:
                    if isinstance(shape, list):
                        product = reduce(lambda x, y: x * y, [dim for dim in shape if isinstance(dim, int)])
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

    # Calculate and print model size
    model_size_bytes = total_params * 4  # Assuming 4 bytes per parameter (float32)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f'Model size: {model_size_mb:.2f} MB')
    print('-------------------------------------------------------')