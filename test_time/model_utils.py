from torch import nn

def toggle_norm_layers(model: nn.Module, freeze: bool = True) -> None:
    """
    Iterates over a PyTorch model to freeze or unfreeze normalization layers.

    Args:
        model (nn.Module): The model containing the layers.
        freeze (bool): If True, freezes the layers (param.requires_grad = False).
                       If False, unfreezes the layers (param.requires_grad = True).
    """
    # A tuple of common normalization layers
    norm_layers = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
    )

    for module in model.modules():
        if isinstance(module, norm_layers):
            # Set requires_grad for all parameters in the module
            for param in module.parameters():
                param.requires_grad = not freeze
