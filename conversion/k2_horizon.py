from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, gguf

@ModelBase.register(
    "K2HorizonForCausalLM",
    "K2AuroraForCausalLM", # TODO: DELETE
)
@ModelBase.example(
    "IFM/K2-Horizon-0.9B",
    "IFM/K2-Horizon-36B",
)
class K2HorizonModel(TextModel):
    model_arch = gguf.MODEL_ARCH.K2HORIZON

    def get_vocab_tokenizer(self):
        from transformers import PreTrainedTokenizerFast

        return PreTrainedTokenizerFast.from_pretrained(
            self.dir_model,
            local_files_only=True,
        )

    def set_vocab(self):
        super().set_vocab()

        template_path = (
            Path(__file__).parent.parent
            / "models"
            / "templates"
            / "k2-horizon.jinja"
        )
        template = template_path.read_text(encoding="utf-8")
        self.gguf_writer.remove_key(gguf.Keys.Tokenizer.CHAT_TEMPLATE)
        self.gguf_writer.add_chat_template(template)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # generic
        rope_head_dim = self.hparams.get("rope_head_dim")
        norm_groups = int(self.hparams.get("layernorm_num_groups", 1))

        self.gguf_writer.add_group_norm_groups(norm_groups)
        if rope_head_dim is not None:
            self.gguf_writer.add_rope_dimension_count(int(rope_head_dim))

        # moe
        num_experts = int(self.hparams.get("num_experts", 0))
        if num_experts > 0:
            moe_ff = int(self.hparams["moe_intermediate_size"])
            dense_layers = self.hparams.get("num_dense_layers")
            mlp_only_layers = {int(layer) for layer in self.hparams.get("mlp_only_layers", [])}
            sparse_step = int(self.hparams.get("decoder_sparse_step", 1))
            shared_experts = int(self.hparams.get("num_shared_experts", 0))
            router_scale = self.hparams.get("router_scaling_factor")
            normalize_topk = bool(self.hparams.get("norm_topk_prob", False))
            router_func = self.hparams.get("router_score_func")

            if dense_layers is None:
                dense_layers = 0
                while dense_layers in mlp_only_layers:
                    dense_layers += 1

            self.gguf_writer.add_expert_feed_forward_length(moe_ff)
            self.gguf_writer.add_leading_dense_block_count(dense_layers)            
            self.gguf_writer.add_moe_every_n_layers(sparse_step)
            self.gguf_writer.add_expert_shared_count(shared_experts)
            self.gguf_writer.add_expert_weights_norm(normalize_topk)
            if shared_experts > 0:
                self.gguf_writer.add_expert_shared_feed_forward_length(moe_ff * shared_experts)
            if router_scale is not None:
                self.gguf_writer.add_expert_weights_scale(float(router_scale))
            match router_func:
                case "sigmoid":
                    gating_func = gguf.ExpertGatingFuncType.SIGMOID    
                case "softmax":
                    gating_func = gguf.ExpertGatingFuncType.SOFTMAX
                case _:
                    raise ValueError(f"Unsupported router_score_func: {router_func!r}")
            self.gguf_writer.add_expert_gating_func(gating_func)

        # mova
        value_experts = int(self.hparams.get("mova_num_experts", 0))
        value_experts_used = int(self.hparams.get("mova_num_experts_per_tok", 0))

        if value_experts > 0 and value_experts_used > 0:
            assert value_experts_used <= value_experts
            self.gguf_writer.add_attention_value_expert_count(value_experts)
            self.gguf_writer.add_attention_value_expert_used_count(value_experts_used)

        # gate func, only making sure it exists and is softplus
        gate_func = self.hparams.get("attention_gate_func")
        if gate_func not in (None, "softplus"):
            raise ValueError(f"Unsupported attention_gate_func: {gate_func!r}")

    _experts: list[dict[str, Tensor]] | None = None
    _value_experts: list[dict[str, Tensor]] | None = None
    def modify_tensors(
        self,
        data_torch: Tensor,
        name: str,
        bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        # MoE: router
        if name.endswith(".mlp.gate.bias"):
            assert bid is not None
            yield (
                self.format_tensor_name(
                    gguf.MODEL_TENSOR.FFN_EXP_PROBS_B,
                    bid,
                    ".bias"
                ),
                data_torch
            )
            return

        # MoE: actual up down or gate
        is_moe_tensor = re.fullmatch(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(down_proj|gate_proj|up_proj)\.weight", name)
        if is_moe_tensor:
            assert bid is not None
            num_experts = int(self.hparams["num_experts"])

            # allocate on first layer that has experts
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            # atp, this_blocks_experts contains all experts
            this_blocks_experts = self._experts[bid]
            this_blocks_experts[name] = data_torch

            # filling up self._experts until up down gate are all inside, then continue
            if len(this_blocks_experts) < num_experts * 3:
                return

            for projection in ("down_proj", "gate_proj", "up_proj"):
                tensors = []
                for expert_id in range(num_experts):
                    expert_name = f"model.layers.{bid}.mlp.experts.{expert_id}.{projection}.weight"
                    tensors.append(this_blocks_experts.pop(expert_name))
                merged = torch.stack(tensors, dim=0)
                merged_name = f"model.layers.{bid}.mlp.experts.{projection}.weight"
                yield from super().modify_tensors(
                    merged,
                    merged_name,
                    bid
                )
            return
        
        # MoVA
        is_mova_weights = re.fullmatch(r"model\.layers\.\d+\.self_attn\.v_experts\.\d+\.weight", name)
        if is_mova_weights:
            assert bid is not None
            num_value_experts = int(self.hparams["mova_num_experts"])
            if self._value_experts is None:
                self._value_experts = [{} for _ in range(self.block_count)]

            this_blocks_value_expert = self._value_experts[bid]
            this_blocks_value_expert[name] = data_torch

            # no need to * 3 because no up down gate like normal moe
            if len(this_blocks_value_expert) < num_value_experts:
                return

            tensors = []
            for value_exp_id in range(num_value_experts):
                value_exp_name = f"model.layers.{bid}.self_attn.v_experts.{value_exp_id}.weight"
                tensors.append(this_blocks_value_expert.pop(value_exp_name))

            merged = torch.stack(tensors, dim = 0)
            merged_name = f"model.layers.{bid}.self_attn.v_experts.weight"
            yield from super().modify_tensors(
                merged,
                merged_name,
                bid
            )
            return

        # fallback, the default way basically
        yield from super().modify_tensors(
            data_torch,
            name,
            bid
        )

    def prepare_tensors(self):
        super().prepare_tensors()

        # this is just checks basically
        if self._experts is not None:
            remaining_experts = [
                name
                for block in self._experts
                for name in block
            ]

            if remaining_experts:
                raise ValueError(
                    f"Unprocessed MoE experts: {remaining_experts}"
                )

        if self._value_experts is not None:
            remaining_value_experts = [
                name
                for block in self._value_experts
                for name in block
            ]

            if remaining_value_experts:
                raise ValueError(
                    "Unprocessed MoVA value experts: "
                    f"{remaining_value_experts}"
                )

