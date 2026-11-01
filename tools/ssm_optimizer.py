"""Optional optimizer protection for Simba's three SSM add-on parameters.

The default returns the original parameter iterator. The opt-in path leaves
all backbone weight decay unchanged, including existing Mamba A_log/D flags.
"""

import math


_ADDON_NAMES = frozenset(('bidir_logit', 'motion_down.weight', 'motion_up.weight'))


def optimizer_parameter_groups(model, weight_decay, protect_ssm_addons=False):
    """Exclude only parameters directly inside ``*.mamba.mamba`` add-ons.

    Pass the return value to the optimizer's ``params`` argument and retain
    its usual global ``weight_decay`` argument. Frozen parameters and shared
    parameter de-duplication follow the model's standard iterators.
    """
    if not protect_ssm_addons:
        return model.parameters()
    decay = float(weight_decay)
    if not math.isfinite(decay) or decay < 0:
        raise ValueError('weight_decay must be finite and non-negative')

    backbone, addons = [], []
    for name, parameter in model.named_parameters():
        _, marker, leaf = name.rpartition('.mamba.mamba.')
        destination = addons if marker and leaf in _ADDON_NAMES else backbone
        destination.append(parameter)
    if not addons:
        raise ValueError('SSM add-on protection requested, but no bidir_logit or '
                         'motion_down/up weights were found inside *.mamba.mamba')

    groups = []
    if backbone:
        groups.append(dict(params=backbone, weight_decay=decay, group_name='backbone'))
    groups.append(dict(params=addons, weight_decay=0.0, group_name='ssm_addons'))
    return groups
