# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DualBigMoE model."""

import copy

import torch
from torch import nn

from ... import initialization as init
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from .configuration_dualbig_moe import DualBigMoEConfig


logger = logging.get_logger(__name__)


class DualBigMoERMSNorm(Qwen3RMSNorm):
    pass


class DualBigMoERotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class DualBigMoEAttention(Qwen3Attention):
    pass


class DualBigMoEBigMLP(Qwen3MLP):
    pass


class DualBigMoEMLP(nn.Module):
    """双大核 + 微专家池的融合 FFN，替换 Qwen3 每层的 ``mlp``。

    前向 = ``w_arts · FFN_arts(x) + w_sci · FFN_sci(x) + gamma · Σ wᵢ · LoRAᵢ(x)``
    """

    def __init__(self, config: DualBigMoEConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.rank = config.lora_rank
        self.gamma = config.gamma
        self.scaling = config.lora_alpha / config.lora_rank
        self.group_sizes = config.group_sizes
        self.group_bounds = config.group_bounds

        # Tier-1 文科锚核：原版 MLP，逐位等于基座，冻结。
        self.big_arts = DualBigMoEBigMLP(config)
        for p in self.big_arts.parameters():
            p.requires_grad = False

        # Tier-2 理科孪生核：克隆一份，低学习率微调。
        self.big_sci = copy.deepcopy(self.big_arts)
        for p in self.big_sci.parameters():
            p.requires_grad = True

        # 双大核路由器（无 bias）与微专家路由器。
        self.router_big = nn.Linear(
            config.hidden_size,
            2,
            bias=False,
            device=self.big_arts.gate_proj.weight.device,
            dtype=self.big_arts.gate_proj.weight.dtype,
        )
        self.router_little = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            device=self.big_arts.gate_proj.weight.device,
            dtype=self.big_arts.gate_proj.weight.dtype,
        )

        # Tier-3 堆叠 LoRA，形状 (E, r, D)。堆叠而非 ModuleList 是为了让所有
        # 专家都能被稠密计算覆盖。
        self.lora_A = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.lora_rank,
                config.hidden_size,
                device=self.big_arts.gate_proj.weight.device,
                dtype=self.big_arts.gate_proj.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(torch.empty_like(self.lora_A))
        self.reset_expert_parameters_()

        # ---- 流式搬运运行时状态（不进 state_dict）----
        self._host_pool: dict[str, torch.Tensor] = {}
        self._pool_key: tuple | None = None
        self._slot = 0
        self.staging_A: list[torch.Tensor] = []
        self.staging_B: list[torch.Tensor] = []
        self.staging_events: list | None = None
        self.transfer_stream = None
        self._telemetry: dict[str, torch.Tensor] | None = None

    @property
    def telemetry(self) -> dict[str, torch.Tensor]:
        """路由统计累加器，按当前设备惰性创建。"""
        if self._telemetry is None or self._telemetry["expert_calls"].device != self.lora_A.device:
            self._telemetry = self._new_telemetry()
        return self._telemetry

    def reset_expert_parameters_(self) -> None:
        """A 用 kaiming、B 置零 —— 初始输出恒为 0，不破坏基座行为。

        已加载权重（``_is_hf_initialized``）时跳过：``from_pretrained`` 在加载后会
        为缺失键再次调用 ``_init_weights``，而 ``lora_A`` 的 ``reshape`` 视图不会
        携带该标志，若无此守卫会用随机值覆盖已加载的 checkpoint。
        """
        if getattr(self.lora_A, "_is_hf_initialized", False) and getattr(self.lora_B, "_is_hf_initialized", False):
            return
        init.kaiming_uniform_(self.lora_A.reshape(-1, self.hidden_size), a=5**0.5)
        init.zeros_(self.lora_B)

    def _new_telemetry(self) -> dict[str, torch.Tensor]:
        device = self.lora_A.device
        return {
            "arts_weight": torch.zeros((), device=device, dtype=torch.float32),
            "sci_weight": torch.zeros((), device=device, dtype=torch.float32),
            "expert_calls": torch.zeros(self.num_experts, device=device, dtype=torch.long),
        }

    def reset_telemetry(self) -> None:
        """清零路由统计。推理脚本在每次提问前调用。"""
        for t in self.telemetry.values():
            t.zero_()

    def _big_core(self, x: torch.Tensor) -> torch.Tensor:
        """双大核逐位置混音：``w_arts · FFN_arts(x) + w_sci · FFN_sci(x)``。"""
        weights = torch.softmax(self.router_big(x).float(), dim=-1).to(x.dtype)
        return weights[..., 0:1] * self.big_arts(x) + weights[..., 1:2] * self.big_sci(x)

    @property
    def pool_location(self) -> str:
        return getattr(self.config, "expert_pool_location", "host")

    def _ensure_expert_pool(self) -> None:
        """确保 Tier-3 权重就位。惰性构建，由 ``forward`` 调用。"""
        device = self.lora_A.device
        key = (self.pool_location, str(device), str(self.lora_A.dtype))
        if self._pool_key == key:
            return
        self._pool_key = key

        if self.pool_location == "device":
            self._host_pool = {}
            self.staging_A, self.staging_B = [], []
            self.staging_events = None
            self.transfer_stream = None
            return

        self._host_pool = {
            "A": self.lora_A.detach().to("cpu", copy=True).contiguous().pin_memory(),
            "B": self.lora_B.detach().to("cpu", copy=True).contiguous().pin_memory(),
        }
        self._setup_staging(device)

    def _setup_staging(self, device: torch.device) -> None:
        """建立常驻 staging buffer 与 ping-pong 完成事件。"""
        n = max(1, int(getattr(self.config, "num_staging_buffers", 2)))
        self.staging_A = [
            torch.empty(self.top_k, self.rank, self.hidden_size, device=device, dtype=self.lora_A.dtype)
            for _ in range(n)
        ]
        self.staging_B = [torch.empty_like(a) for a in self.staging_A]
        self._slot = 0
        self.staging_events = None
        self.transfer_stream = None

        if device.type == "cuda" and torch.cuda.is_available():
            self.staging_events = [torch.cuda.Event() for _ in range(n)]
            current = torch.cuda.current_stream(device)
            for ev in self.staging_events:
                ev.record(current)
            self.transfer_stream = torch.cuda.Stream(device=device)

    def _stage_experts(self, selected: list[int], device: torch.device) -> int:
        """把 Top-k 专家搬入 staging buffer，返回本拍槽位。"""
        slot = self._slot
        self._slot = (self._slot + 1) % len(self.staging_A)
        host_a, host_b = self._host_pool["A"], self._host_pool["B"]

        if self.transfer_stream is None:
            for j, eid in enumerate(selected):
                self.staging_A[slot][j].copy_(host_a[eid])
                self.staging_B[slot][j].copy_(host_b[eid])
            return slot

        self.transfer_stream.wait_event(self.staging_events[slot])
        with torch.cuda.stream(self.transfer_stream):
            for j, eid in enumerate(selected):
                self.staging_A[slot][j].copy_(host_a[eid], non_blocking=True)
                self.staging_B[slot][j].copy_(host_b[eid], non_blocking=True)
        torch.cuda.current_stream(device).wait_stream(self.transfer_stream)
        return slot

    def _compute_experts(self, x: torch.Tensor, weights: list[float], slot: int) -> torch.Tensor:
        """在 staging buffer 上做 ``sum_i w_i · B_i(A_i x)``。"""
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen
        a_buf, b_buf = self.staging_A[slot], self.staging_B[slot]

        x2d = x.reshape(n, dim)
        out = torch.zeros(n, dim, device=x.device, dtype=x.dtype)
        for j, w in enumerate(weights):
            h = torch.matmul(x2d, a_buf[j].t())
            out.addmm_(h, b_buf[j], alpha=w)

        if self.staging_events is not None:
            self.staging_events[slot].record(torch.cuda.current_stream(x.device))
        return out.reshape(bsz, seqlen, dim) * self.scaling

    def _compute_experts_resident(self, x: torch.Tensor, selected: list[int], weights: list[float]) -> torch.Tensor:
        """零拷贝路径：直接从常驻参数读选中的专家，省掉 PCIe 往返。"""
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen
        x2d = x.reshape(n, dim)
        out = torch.zeros(n, dim, device=x.device, dtype=x.dtype)
        for j, eid in enumerate(selected):
            h = torch.matmul(x2d, self.lora_A[eid].t())
            out.addmm_(h, self.lora_B[eid], alpha=weights[j])
        return out.reshape(bsz, seqlen, dim) * self.scaling

    def _route_little(self, x: torch.Tensor):
        """用末 token 选专家，返回归一化后的 (下标, 权重)。"""
        last_tok = x[:, -1:, :]
        probs = torch.softmax(self.router_little(last_tok).float(), dim=-1).to(x.dtype)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        return topi, topv

    def _record_telemetry(self, x: torch.Tensor, topi, topv) -> None:
        """路由统计全部在 GPU 上累加，避免每层每 token 的 ``.item()`` 同步。"""
        t = self.telemetry
        with torch.no_grad():
            w = torch.softmax(self.router_big(x[:, -1:, :]).float(), dim=-1)
            t["arts_weight"] += w[0, -1, 0].float()
            t["sci_weight"] += w[0, -1, 1].float()
            t["expert_calls"].index_add_(0, topi[0, -1], torch.ones_like(topi[0, -1], device=t["expert_calls"].device))

    def expert_output(self, x: torch.Tensor) -> torch.Tensor:
        """Tier-3 微专家对激活的贡献（已含 LoRA scaling，不含 gamma）。"""
        topi, topv = self._route_little(x)
        self._record_telemetry(x, topi, topv)

        info = torch.stack([topi[0, -1].to(x.dtype), topv[0, -1]]).to("cpu").tolist()
        selected = [int(v) for v in info[0]]
        weights = [float(v) for v in info[1]]

        if self.pool_location == "device":
            return self._compute_experts_resident(x, selected, weights)

        self._ensure_expert_pool()
        slot = self._stage_experts(selected, x.device)
        return self._compute_experts(x, weights, slot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        big_out = self._big_core(x)
        return big_out + self.gamma * self.expert_output(x)

    def extra_repr(self) -> str:
        return (
            f"experts={self.num_experts}, top_k={self.top_k}, rank={self.rank}, "
            f"gamma={self.gamma}, pool={self.pool_location}"
        )


class DualBigMoEDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: DualBigMoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = DualBigMoEMLP(config, layer_idx)


class DualBigMoEPreTrainedModel(Qwen3PreTrainedModel):
    config: DualBigMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DualBigMoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DualBigMoEDecoderLayer,
        "attentions": DualBigMoEAttention,
    }

    def _init_weights(self, module: nn.Module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DualBigMoEMLP):
            module.reset_expert_parameters_()
            for proj in (module.big_sci.gate_proj, module.big_sci.up_proj, module.big_sci.down_proj):
                init.normal_(proj.weight, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.router_big.weight, std=self.config.initializer_range)
            init.normal_(module.router_little.weight, std=self.config.initializer_range)


class DualBigMoEModel(Qwen3Model):
    pass


class DualBigMoEForCausalLM(Qwen3ForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    config: DualBigMoEConfig

    def moe_layers(self) -> list[DualBigMoEMLP]:
        """取出全部双大核 FFN 模块。"""
        return [layer.mlp for layer in self.model.layers if isinstance(layer.mlp, DualBigMoEMLP)]

    def reset_routing_stats(self) -> None:
        """清零全部层的路由统计。"""
        for m in self.moe_layers():
            m.reset_telemetry()

    @torch.no_grad()
    def routing_report(self) -> dict:
        """一次性取回全部遥测统计（避免每层各自同步）。"""
        modules = self.moe_layers()
        if not modules:
            return {}
        arts = torch.stack([m.telemetry["arts_weight"] for m in modules]).sum().item()
        sci = torch.stack([m.telemetry["sci_weight"] for m in modules]).sum().item()
        total = arts + sci
        calls = torch.stack([m.telemetry["expert_calls"] for m in modules]).sum(0).cpu().tolist()
        grand = float(sum(calls))
        bounds = self.config.group_bounds
        shares = [(sum(calls[lo:hi]) / grand if grand > 0 else 0.0) for lo, hi in bounds]
        return {
            "arts_ratio": arts / total if total > 0 else 0.5,
            "sci_ratio": sci / total if total > 0 else 0.5,
            "expert_calls": calls,
            "expert_share": [c / grand for c in calls] if grand > 0 else [0.0] * len(calls),
            "group_bounds": bounds,
            "group_shares": shares,
            "group_fair_share": [(hi - lo) / self.config.num_experts for lo, hi in bounds],
        }

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        **super_kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, DualBigMoEForCausalLM

        >>> model = DualBigMoEForCausalLM.from_pretrained("your-org/dualbig-qwen3-0.6b")
        >>> tokenizer = AutoTokenizer.from_pretrained("your-org/dualbig-qwen3-0.6b")

        >>> prompt = "解释一下快速排序"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "解释一下快速排序：快速排序是一种分治排序算法..."
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "DualBigMoEForCausalLM",
    "DualBigMoEModel",
    "DualBigMoEPreTrainedModel",
]
