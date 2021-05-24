import torch


def calc_num_elements(module, module_input_shape, format="NCHW"):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    if format == "NHWC":
        some_input = some_input.permute(0, 3, 1, 2)
    num_elements = module(some_input).numel()
    return num_elements


def to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value
