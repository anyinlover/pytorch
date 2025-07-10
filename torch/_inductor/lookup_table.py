"""
Template Lookup Table System

This module provides a lookup table system for template configurations used in autotuning.
The typical usage pattern is:

1. Get the table for your operation:
   op_lookup_dict = lookup_op_configs_by_template_id(input_nodes, op_name)

2. Get the configs for your template (if there's support for it):
   template_params = lookup_op_configs_for_template_id(op_lookup_dict, template_id)

3. Use those configs to append choices directly:
   for kwargs in template_params:
       template.maybe_append_choice(choices, input_nodes=input_nodes, **kwargs)

4. Extract final choices (handles ATEN fallback when using lookup table):
   choices, using_aten = lookup_table_extract_choice(choices)

Note: None is used to indicate that the table is not in use, whereas empty lists and
dicts are fine to use (they indicate no matching configs were found).

See torch/_inductor/kernel/mm.py (tuned_mm function) for a complete usage example.
"""

import logging
from typing import Any, Optional

import torch
from torch._inductor.virtualized import V

from . import config as inductor_config


log = logging.getLogger(__name__)


def _in_use() -> bool:
    """
    Determine if the template lookup table should be used.
    This system is only used when cuda is available and the table has data.
    """
    active = torch.cuda.is_available() and bool(
        inductor_config.template_lookup_table.table
    )
    # the lookup table requires max-autotune and it is an error to use it without max-autotune
    if active and not (
        inductor_config.max_autotune or inductor_config.max_autotune_gemm
    ):
        raise RuntimeError(
            "The template lookup table requires max-autotune to be enabled. "
            "Please set inductor_config.max_autotune=True, or remove the lookup table. "
        )
    return active


def _get_lookup_table() -> Optional[
    dict[str, dict[str, dict[str, list[dict[str, Any]]]]]
]:
    """
    Get the template lookup table from config.
    """
    if not _in_use():
        return None
    return inductor_config.template_lookup_table.table


def _dev_key(device: torch.device) -> str:
    """
    Generate a device key for lookup table indexing.
    For CPU devices, raises an error.
    For CUDA devices, returns a string combining device name and capability.
    """
    if device.type != "cuda":
        raise ValueError("CPU devices are not supported for lookup table")

    # Get CUDA device properties and capability
    props = torch.cuda.get_device_properties(device.index)
    capability = torch.cuda.get_device_capability(device.index)

    # Return string combining gcnArchName and capability
    return f"{props.gcnArchName}{capability}"


def _template_lookup_key(input_nodes: list[Any]) -> str:
    return str(
        tuple(
            # List here, because we want a standard encoding e.g. [] instead of ()
            (node.get_dtype(), list(get_size_hint(node)), list(get_stride_hint(node)))
            for node in input_nodes
        )
    )


def _get_op_lookup_table(
    input_nodes: list[Any], op: str
) -> Optional[list[dict[str, Any]]]:
    """
    Get the lookup table configs for a specific operation and input configuration.

    Args:
        input_nodes: List of input nodes for the operation
        op: Operation name (e.g., "mm", "addmm")

    Returns:
        List of complete template_options dictionaries, or None if not found
    """
    lookup_table = _get_lookup_table()
    if not _in_use() or lookup_table is None:
        return None

    # Assume the first input parameter is used to determine the device
    device = input_nodes[0].get_device()
    dev_key = _dev_key(device)
    log.debug("device_name: %s", dev_key)

    # Generate lookup table key
    lookup_key = _template_lookup_key(input_nodes)
    log.debug("lookup_key: %s", lookup_key)

    # Retrieve the config list directly
    # Since we know the system is in use, if there is no match, we just return an empty list
    config_list = lookup_table.get(dev_key, {}).get(op, {}).get(lookup_key, [])

    if config_list is not None:
        # Validate that each config has template_id
        for i, config in enumerate(config_list):
            if not isinstance(config, dict):
                raise ValueError(
                    f"Config at index {i} for {op} operation is not a dictionary: {config}"
                )
            if "template_id" not in config:
                raise ValueError(
                    f"Config at index {i} for {op} operation missing required 'template_id' field: {config}"
                )

    log.debug("config_list for %s: %s", op, config_list)
    return config_list


def lookup_op_configs_by_template_id(
    input_nodes: list[Any], op: str
) -> Optional[dict[str, list[dict[str, Any]]]]:
    """
    Get configs grouped by template_id for a specific operation and input configuration.

    Args:
        input_nodes: List of input nodes for the operation
        op: Operation name (e.g., "mm", "addmm")

    Returns:
        Dictionary mapping template_id to lists of configs, or None if not found
    """
    from collections import defaultdict

    config_list = _get_op_lookup_table(input_nodes, op)
    if config_list is None:
        return None

    # Group configs by template_id
    grouped_configs = defaultdict(list)
    for config in config_list:
        template_id = config["template_id"]
        grouped_configs[template_id].append(config)

    return dict(grouped_configs)


def lookup_table_extract_choice(choices: list[Any]) -> tuple[list[Any], bool]:
    """
    If there are multiple choices, this means that the lookup table was used.
    The initial choice is always ATEN, so we want to skip it and return the rest,
    if there are other choices
    """
    if not _in_use():
        return choices, False
    if len(choices) > 1:
        # We want to skip the ATEN choice and return the rest
        return choices[1:], False

    from torch._inductor.select_algorithm import ExternKernelCaller

    # indicate to the caller that we're using the ATEN choice
    # as they might use this information to modify the layout
    return choices, isinstance(choices[0], ExternKernelCaller)


def get_size_hint(mat: Any) -> list[int]:
    size = mat.get_size()
    if not all(isinstance(dim, int) for dim in size):
        size = V.graph.sizevars.size_hints(
            size,
            fallback=inductor_config.unbacked_symint_fallback,
        )
    return list(size)


def get_stride_hint(mat: Any) -> list[int]:
    stride = mat.get_stride()
    if not all(isinstance(dim, int) for dim in stride):
        stride = V.graph.sizevars.size_hints(
            stride,
            fallback=inductor_config.unbacked_symint_fallback,
        )
    return list(stride)


def lookup_op_configs_for_template_id(
    lookup_dict: Optional[dict[str, list[dict[str, Any]]]], template_id: str
) -> Optional[list[dict[str, Any]]]:
    """
    Look up and filter template configurations for a specific template_id.

    This consolidates functionality from get_template_params and lookup_template_dict,
    and includes TF32 filtering logic.

    Args:
        lookup_dict: Dictionary from lookup_op_configs_by_template_id (may be None)
        template_id: Template identifier (e.g., "triton", "tma", "bias_addmm")

    Returns:
        None: No lookup table is in use or lookup_dict is None
        []: Lookup table exists but no match found or all configs filtered out
        [kwargs1, kwargs2, ...]: Match found, filtered configurations
    """
    if not _in_use():
        return None

    # If lookup_dict is None
    if lookup_dict is None:
        return None

    # If no match found, return empty list, as we don't have any configs
    configs = lookup_dict.get(template_id, [])

    # Filter out configs with ALLOW_TF32=True when torch.backends.cuda.matmul.allow_tf32 is False
    filtered_configs = []
    for config in configs:
        if (
            "ALLOW_TF32" in config
            and config["ALLOW_TF32"] is True
            and not torch.backends.cuda.matmul.allow_tf32
        ):
            log.warning(
                "Filtering out config with ALLOW_TF32=True because "
                "torch.backends.cuda.matmul.allow_tf32 is False. Config: %s",
                config,
            )
            continue
        # Lastly, we have to throw out the template_id, as it's not a valid kwarg
        # and just used to identify which template the entry belongs to
        del config["template_id"]
        filtered_configs.append(config)

    return filtered_configs
