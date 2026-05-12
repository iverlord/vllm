# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class MTPProposer(SpecDecodeBaseProposer):
    """
    Proposer for Multi-Token Prediction (MTP) models.
    
    MTP models add extra layers after the main model to predict multiple tokens
    in a single forward pass. When using pipeline parallelism, MTP layers must
    be executed on the last pipeline rank (where lm_head resides).
    
    This proposer ensures that:
    1. MTP layers are only instantiated on the last pipeline rank
    2. Hidden states are properly transferred between pipeline ranks
    3. The MTP computation happens on the last GPU in the pipeline
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # Check if we're on the last PP rank - MTP only works there
        if not get_pp_group().is_last_rank:
            raise RuntimeError(
                "MTPProposer can only be instantiated on the last pipeline parallel rank. "
                f"Current rank is_first={get_pp_group().is_first_rank}, "
                f"is_last={get_pp_group().is_last_rank}, "
                f"world_size={get_pp_group().world_size}"
            )
        
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
