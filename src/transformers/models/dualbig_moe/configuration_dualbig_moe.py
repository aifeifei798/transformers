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
"""DualBigMoE model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen3-0.6B-Base")
@strict(accept_kwargs=True)
class DualBigMoEConfig(PreTrainedConfig):
    r"""
    num_experts (`int`, *optional*, defaults to 32):
        Tier-3 每层微专家数。
    top_k_experts (`int`, *optional*, defaults to 8):
        每 token 激活的微专家数。刻意不叫 `top_k` 以避免与 `GenerationConfig`  colliding。
    lora_rank (`int`, *optional*, defaults to 16):
        堆叠 LoRA 权重 `lora_A` / `lora_B` 的秩，形状为 `(E, rank, hidden)`。
    lora_alpha (`float`, *optional*, defaults to 16.0):
        LoRA 缩放分子的 `alpha`，实际缩放为 `alpha / rank`。
    gamma (`float`, *optional*, defaults to 0.3):
        微专家对大核输出的缩放系数。
    group_sizes (`list[int]`, *optional*):
        专家分组边界（连续区间，如代码 / 数学 / 写作），默认为 8/8/16。
    expert_pool_location (`str`, *optional*, defaults to `"host"`):
        Tier-3 专家池的存放位置。`"host"` 走 pinned 主机内存 + PCIe 流式搬运；
        `"device"` 常驻显存，跳过搬运，数值完全等价。
    num_staging_buffers (`int`, *optional*, defaults to 2):
        流式搬运的 staging buffer 槽数。2 = 双缓冲，拷贝与计算真正重叠。

    ```python
    >>> from transformers import DualBigMoEModel, DualBigMoEConfig

    >>> # Initializing a DualBigMoE style configuration
    >>> configuration = DualBigMoEConfig()

    >>> # Initializing a model from the configuration
    >>> model = DualBigMoEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dualbig_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `DualBigMoE`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.big_arts.gate_proj": "colwise",
        "layers.*.mlp.big_arts.up_proj": "colwise",
        "layers.*.mlp.big_arts.down_proj": "rowwise",
        "layers.*.mlp.big_sci.gate_proj": "colwise",
        "layers.*.mlp.big_sci.up_proj": "colwise",
        "layers.*.mlp.big_sci.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151645
    # ---- 双大核 + 微专家池 ----
    num_experts: int = 32
    top_k_experts: int = 8
    lora_rank: int = 16
    lora_alpha: float = 16.0
    gamma: float = 0.3
    group_sizes: list[int] | None = None
    expert_pool_location: str = "host"
    num_staging_buffers: int = 2

    def __post_init__(self, **kwargs):
        # rope 统一走 rope_parameters 字典（transformers 5.x 约定）。
        # 保留 rope_theta 顶层键以兼容仍读旧字段的下游代码。
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": float(kwargs.pop("rope_theta", 1000000.0)),
            }

        # 兼容手写 config.json 里遗留的 ``top_k``：它曾是本模型的专家 Top-k，
        # 但对 GenerationConfig 而言是采样参数。显式搬过来并丢弃，避免
        # setattr 撞上只读 property 直接抛 AttributeError。
        # 刻意不叫 ``top_k``，也不提供 ``top_k`` 别名：``top_k`` 是
        # ``GenerationConfig`` 的字段名，``PreTrainedModel.__init__`` 会把它
        # 扫进生成配置，静默改变每一次 generate() 的输出分布。
        legacy_top_k = kwargs.pop("top_k", None)
        if legacy_top_k is not None:
            self.top_k_experts = legacy_top_k

        if self.group_sizes is None:
            # 默认 8/8/16 = 代码 / 数学 / 写作
            self.group_sizes = self._default_group_sizes(self.num_experts)
        self.group_sizes = [int(s) for s in self.group_sizes]

        # Qwen3 的 GQA 与滑动窗口约定：KV 头缺省等于 Q 头；未启用滑动窗口时
        # 视为全层 full_attention，缓存与 mask 都依赖这个归一化结果。
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if not self.use_sliding_window:
            self.sliding_window = None
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)
        self._validate()

    @staticmethod
    def _default_group_sizes(num_experts: int) -> list[int]:
        """把 32 路专家按 8/8/16 切成三组；其他专家数按等比缩放。"""
        if num_experts == 32:
            return [8, 8, 16]
        if num_experts <= 0:
            return [num_experts]
        third = max(1, num_experts // 4)
        half = max(1, (num_experts - third) // 2)
        return [third, half, num_experts - third - half]

    def _validate(self) -> None:
        """把结构性错误挡在构造期，而不是留到生成时数值错位。"""
        if self.num_experts <= 0:
            raise ValueError(f"num_experts 必须为正，实际 {self.num_experts}")
        if not 0 < self.top_k_experts <= self.num_experts:
            raise ValueError(f"top_k_experts 必须落在 1..{self.num_experts}，实际 {self.top_k_experts}")
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank 必须为正，实际 {self.lora_rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha 必须为正，实际 {self.lora_alpha}")
        if self.gamma < 0:
            raise ValueError(f"gamma 不能为负，实际 {self.gamma}")
        if self.num_staging_buffers <= 0:
            raise ValueError(f"num_staging_buffers 必须为正，实际 {self.num_staging_buffers}")
        if self.expert_pool_location not in ("host", "device"):
            raise ValueError(f"expert_pool_location 只能是 'host' 或 'device'，实际 {self.expert_pool_location!r}")
        total = sum(self.group_sizes)
        if total != self.num_experts:
            raise ValueError(f"group_sizes 之和 {total} 与 num_experts {self.num_experts} 不符")
        if any(s <= 0 for s in self.group_sizes):
            raise ValueError(f"group_sizes 不允许空组：{self.group_sizes}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types 长度 {len(self.layer_types)} 与 num_hidden_layers {self.num_hidden_layers} 不符"
            )

    @property
    def lora_scaling(self) -> float:
        """LoRA 缩放系数 ``alpha / rank``。"""
        return self.lora_alpha / self.lora_rank

    @property
    def group_bounds(self) -> list[tuple[int, int]]:
        """把 ``group_sizes`` 展开为连续半开区间，供分组级路由与遥测使用。"""
        bounds, cur = [], 0
        for size in self.group_sizes:
            bounds.append((cur, cur + size))
            cur += size
        return bounds


__all__ = ["DualBigMoEConfig"]
