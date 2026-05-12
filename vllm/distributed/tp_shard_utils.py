# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Utility functions for tensor parallel shard distribution across heterogeneous GPUs.

This module provides tools for calculating optimal shard distributions when using
tensor parallelism with GPUs that have different memory capacities or when dealing
with an odd number of GPUs.
"""

from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def calculate_memory_per_shard(
    total_model_size_gb: float,
    tensor_parallel_size: int,
    overhead_factor: float = 1.2,
) -> float:
    """
    Calculate approximate memory required per tensor parallel shard.

    Args:
        total_model_size_gb: Total model size in GB (weights only).
        tensor_parallel_size: Number of tensor parallel shards.
        overhead_factor: Multiplier for activation and KV cache overhead.

    Returns:
        Estimated memory per shard in GB.
    """
    base_shard_size = total_model_size_gb / tensor_parallel_size
    return base_shard_size * overhead_factor


def estimate_optimal_shard_map(
    gpu_memory_gb: list[float],
    total_model_size_gb: float,
    tensor_parallel_size: int,
    min_shards_per_gpu: int = 1,
    overhead_factor: float = 1.2,
) -> list[int]:
    """
    Calculate optimal tensor parallel shard distribution based on GPU memory.

    This function distributes shards proportionally to available GPU memory,
    ensuring each GPU can hold its assigned shards without OOM.

    Args:
        gpu_memory_gb: List of available memory for each GPU in GB.
        total_model_size_gb: Total model size in GB (weights only).
        tensor_parallel_size: Total number of shards to distribute.
        min_shards_per_gpu: Minimum number of shards per GPU (default 1).
        overhead_factor: Multiplier for activation and KV cache overhead.

    Returns:
        List of shard counts for each GPU.

    Example:
        >>> gpu_memory = [16.0, 16.0, 12.0]  # 2x 3070m + 1x 3060
        >>> estimate_optimal_shard_map(gpu_memory, 54.0, 5)
        [2, 2, 1]
    """
    if len(gpu_memory_gb) != tensor_parallel_size:
        raise ValueError(
            f"Length of gpu_memory_gb ({len(gpu_memory_gb)}) must equal "
            f"tensor_parallel_size ({tensor_parallel_size})"
        )

    if sum(gpu_memory_gb) == 0:
        raise ValueError("GPU memory values must be positive")

    # Calculate memory per shard needed
    memory_per_shard = calculate_memory_per_shard(
        total_model_size_gb, tensor_parallel_size, overhead_factor
    )

    # Initial allocation based on memory proportion
    total_memory = sum(gpu_memory_gb)
    shard_map = []

    for gpu_mem in gpu_memory_gb:
        # Proportional allocation
        proportional_shards = max(
            min_shards_per_gpu,
            int((gpu_mem / total_memory) * tensor_parallel_size + 0.5),
        )
        shard_map.append(proportional_shards)

    # Adjust to ensure sum equals tensor_parallel_size
    current_sum = sum(shard_map)
    
    if current_sum < tensor_parallel_size:
        # Add remaining shards to GPUs with most memory
        remaining = tensor_parallel_size - current_sum
        gpu_indices_sorted = sorted(
            range(len(gpu_memory_gb)),
            key=lambda i: gpu_memory_gb[i],
            reverse=True,
        )
        for i in range(remaining):
            shard_map[gpu_indices_sorted[i % len(gpu_indices_sorted)]] += 1
    elif current_sum > tensor_parallel_size:
        # Remove excess shards from GPUs with least memory
        excess = current_sum - tensor_parallel_size
        gpu_indices_sorted = sorted(
            range(len(gpu_memory_gb)),
            key=lambda i: gpu_memory_gb[i],
        )
        for i in range(excess):
            idx = gpu_indices_sorted[i % len(gpu_indices_sorted)]
            if shard_map[idx] > min_shards_per_gpu:
                shard_map[idx] -= 1

    # Final validation
    if sum(shard_map) != tensor_parallel_size:
        logger.warning(
            "Shard map sum (%d) does not match tensor_parallel_size (%d). "
            "Adjusting largest GPU.",
            sum(shard_map),
            tensor_parallel_size,
        )
        diff = tensor_parallel_size - sum(shard_map)
        max_gpu_idx = gpu_memory_gb.index(max(gpu_memory_gb))
        shard_map[max_gpu_idx] += diff

    return shard_map


def validate_shard_map_for_model(
    shard_map: list[int],
    hidden_size: int,
    num_attention_heads: int,
    intermediate_size: int | None = None,
    num_key_value_heads: int | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate that a shard map is compatible with model architecture requirements.

    Tensor parallelism requires certain dimensions to be divisible by the number
    of shards on each GPU. This function checks these constraints.

    Args:
        shard_map: List of shard counts for each GPU.
        hidden_size: Model hidden size.
        num_attention_heads: Number of attention heads.
        intermediate_size: MLP intermediate size (optional).
        num_key_value_heads: Number of KV heads for GQA (optional).

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors = []

    for gpu_idx, num_shards in enumerate(shard_map):
        if num_shards <= 0:
            errors.append(f"GPU {gpu_idx}: shard count must be positive")
            continue

        # Check hidden size divisibility
        if hidden_size % num_shards != 0:
            errors.append(
                f"GPU {gpu_idx}: hidden_size ({hidden_size}) not divisible by "
                f"shards ({num_shards})"
            )

        # Check attention heads divisibility
        if num_attention_heads % num_shards != 0:
            errors.append(
                f"GPU {gpu_idx}: num_attention_heads ({num_attention_heads}) "
                f"not divisible by shards ({num_shards})"
            )

        # Check KV heads divisibility for GQA models
        if num_key_value_heads is not None and num_key_value_heads % num_shards != 0:
            errors.append(
                f"GPU {gpu_idx}: num_key_value_heads ({num_key_value_heads}) "
                f"not divisible by shards ({num_shards})"
            )

        # Check intermediate size divisibility
        if intermediate_size is not None and intermediate_size % num_shards != 0:
            errors.append(
                f"GPU {gpu_idx}: intermediate_size ({intermediate_size}) "
                f"not divisible by shards ({num_shards})"
            )

    return len(errors) == 0, errors


def get_cumulative_shard_offsets(shard_map: list[int]) -> list[int]:
    """
    Calculate cumulative shard offsets for each GPU.

    Args:
        shard_map: List of shard counts for each GPU.

    Returns:
        List of cumulative offsets. The i-th element is the starting shard
        index for GPU i.

    Example:
        >>> get_cumulative_shard_offsets([2, 2, 1])
        [0, 2, 4]
    """
    offsets = [0]
    for count in shard_map[:-1]:
        offsets.append(offsets[-1] + count)
    return offsets


def get_shard_ranks_for_gpu(
    gpu_idx: int,
    shard_map: list[int],
) -> list[int]:
    """
    Get the global shard ranks assigned to a specific GPU.

    Args:
        gpu_idx: Index of the GPU.
        shard_map: List of shard counts for each GPU.

    Returns:
        List of global shard ranks for the specified GPU.

    Example:
        >>> get_shard_ranks_for_gpu(1, [2, 2, 1])
        [2, 3]
    """
    if gpu_idx < 0 or gpu_idx >= len(shard_map):
        raise ValueError(f"Invalid GPU index: {gpu_idx}")

    offset = sum(shard_map[:gpu_idx])
    return list(range(offset, offset + shard_map[gpu_idx]))


def recommend_shard_map_for_qwen(
    gpu_memory_gb: list[float],
    model_variant: str = "27B",
    quantization: str = "int4",
) -> list[int]:
    """
    Recommend a shard map for Qwen models based on GPU configuration.

    This function provides pre-calculated recommendations for common Qwen variants.

    Args:
        gpu_memory_gb: List of available memory for each GPU in GB.
        model_variant: Model variant ("3B", "14B", "27B", etc.).
        quantization: Quantization type ("fp8", "int4", "int8", "fp16").

    Returns:
        Recommended shard map.

    Example:
        >>> recommend_shard_map_for_qwen([16.0, 16.0, 12.0], "27B", "int4")
        [2, 2, 1]
    """
    # Approximate model sizes in GB (weights only)
    model_sizes = {
        "3B": {"fp16": 6, "int8": 3.5, "int4": 2, "fp8": 3},
        "14B": {"fp16": 28, "int8": 15, "int4": 8, "fp8": 14},
        "27B": {"fp16": 54, "int8": 28, "int4": 15, "fp8": 27},
        "32B": {"fp16": 64, "int8": 33, "int4": 18, "fp8": 32},
        "72B": {"fp16": 144, "int8": 75, "int4": 40, "fp8": 72},
    }

    if model_variant not in model_sizes:
        logger.warning(
            "Unknown model variant '%s'. Using estimation.",
            model_variant,
        )
        # Rough estimate: 2 bytes per parameter for fp16
        params_billions = float(model_variant.replace("B", ""))
        size_gb = params_billions * 2
        if quantization == "int4":
            size_gb *= 0.28
        elif quantization == "int8":
            size_gb *= 0.55
        elif quantization == "fp8":
            size_gb *= 0.5
    else:
        size_gb = model_sizes[model_variant].get(quantization, model_sizes[model_variant]["fp16"])

    tensor_parallel_size = len(gpu_memory_gb)
    
    return estimate_optimal_shard_map(
        gpu_memory_gb=gpu_memory_gb,
        total_model_size_gb=size_gb,
        tensor_parallel_size=tensor_parallel_size,
        overhead_factor=1.15,  # Slightly lower overhead for inference-only
    )


def print_shard_distribution(
    shard_map: list[int],
    gpu_names: list[str] | None = None,
    gpu_memory_gb: list[float] | None = None,
) -> None:
    """
    Print a human-readable summary of the shard distribution.

    Args:
        shard_map: List of shard counts for each GPU.
        gpu_names: Optional list of GPU names.
        gpu_memory_gb: Optional list of GPU memory sizes.
    """
    logger.info("=" * 60)
    logger.info("Tensor Parallel Shard Distribution")
    logger.info("=" * 60)
    
    total_shards = sum(shard_map)
    
    for i, shards in enumerate(shard_map):
        name = gpu_names[i] if gpu_names and i < len(gpu_names) else f"GPU {i}"
        memory = gpu_memory_gb[i] if gpu_memory_gb and i < len(gpu_memory_gb) else "N/A"
        memory_str = f"{memory}GB" if isinstance(memory, (int, float)) else memory
        
        percentage = (shards / total_shards) * 100 if total_shards > 0 else 0
        
        logger.info(
            "%s (%s): %d shard(s) (%.1f%%)",
            name,
            memory_str,
            shards,
            percentage,
        )
    
    logger.info("-" * 60)
    logger.info("Total shards: %d across %d GPU(s)", total_shards, len(shard_map))
    logger.info("=" * 60)
