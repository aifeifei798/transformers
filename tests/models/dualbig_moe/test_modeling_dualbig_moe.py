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
"""Testing suite for the PyTorch DualBigMoE model."""

import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        DualBigMoEConfig,
        DualBigMoEForCausalLM,
        DualBigMoEModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class DualBigMoEModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DualBigMoEModel

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            attention_dropout=0.0,
            num_experts=8,
            top_k_experts=2,
            lora_rank=4,
            lora_alpha=8.0,
            gamma=0.3,
            group_sizes=[2, 3, 3],
            expert_pool_location="device",
            num_staging_buffers=2,
        )


@require_torch
class DualBigMoEModelTest(CausalLMModelTest, unittest.TestCase):
    # 推理路径用末 token Top-k 经 `.tolist()` 取下标，梯度不回流到 router_little；
    # 与 Qwen3Moe 相同，不要求所有参数都有梯度。
    test_all_params_have_gradient = False
    model_tester_class = DualBigMoEModelTester

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True


@require_torch
class DualBigMoEUnitTest(unittest.TestCase):
    def get_tiny_config(self, **overrides):
        kwargs = {
            "vocab_size": 256,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "num_experts": 8,
            "top_k_experts": 2,
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "gamma": 0.3,
            "group_sizes": [2, 3, 3],
            "expert_pool_location": "device",
        }
        kwargs.update(overrides)
        return DualBigMoEConfig(**kwargs)

    def test_config_defaults(self):
        config = DualBigMoEConfig()
        self.assertEqual(config.model_type, "dualbig_moe")
        self.assertEqual((config.num_experts, config.top_k_experts), (32, 8))
        self.assertEqual(config.group_sizes, [8, 8, 16])
        self.assertAlmostEqual(config.lora_scaling, 1.0)
        # 刻意没有 top_k：避免污染 GenerationConfig
        self.assertFalse(hasattr(config, "top_k"))

    def test_config_legacy_top_k(self):
        config = DualBigMoEConfig(num_experts=8, group_sizes=[2, 3, 3], top_k=3)
        self.assertEqual(config.top_k_experts, 3)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            DualBigMoEConfig(num_experts=0)
        with self.assertRaises(ValueError):
            DualBigMoEConfig(num_experts=8, group_sizes=[2, 3, 3], top_k_experts=9)
        with self.assertRaises(ValueError):
            DualBigMoEConfig(num_experts=8, group_sizes=[2, 2, 2])
        with self.assertRaises(ValueError):
            DualBigMoEConfig(expert_pool_location="hbm")

    def test_auto_registration(self):
        config = self.get_tiny_config()
        self.assertIsInstance(AutoConfig.for_model("dualbig_moe"), DualBigMoEConfig)
        model = AutoModelForCausalLM.from_config(config)
        self.assertIsInstance(model, DualBigMoEForCausalLM)
        base = AutoModel.from_config(config)
        self.assertIsInstance(base, DualBigMoEModel)

    def test_generation_config_not_polluted(self):
        from transformers.generation.configuration_utils import GenerationConfig

        config = self.get_tiny_config()
        gen = GenerationConfig.from_model_config(config)
        # 专家 top-k=2 绝不能变成采样 top-k；默认采样 top-k 应保持 None（v5 默认）
        self.assertNotEqual(gen.top_k, config.top_k_experts)
        self.assertIsNone(gen.top_k)

    def test_forward_device_pool(self):
        config = self.get_tiny_config()
        model = DualBigMoEForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        with torch.no_grad():
            out = model(input_ids)
        self.assertEqual(out.logits.shape, (2, 8, config.vocab_size))

    def test_host_and_device_pool_agree(self):
        config_device = self.get_tiny_config(expert_pool_location="device")
        model = DualBigMoEForCausalLM(config_device).eval()
        # 同一权重下 host 路径必须逐位一致
        for m in model.moe_layers():
            m.config.expert_pool_location = "host"
            m._pool_key = None
        input_ids = torch.randint(0, config_device.vocab_size, (1, 6))
        with torch.no_grad():
            out_host = model(input_ids).logits
        for m in model.moe_layers():
            m.config.expert_pool_location = "device"
            m._pool_key = None
        with torch.no_grad():
            out_device = model(input_ids).logits
        torch.testing.assert_close(out_host, out_device, rtol=1e-4, atol=1e-4)

    def test_save_load_roundtrip(self):
        config = self.get_tiny_config()
        model = DualBigMoEForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 6))
        with torch.no_grad():
            expected = model(input_ids).logits
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            reloaded = DualBigMoEForCausalLM.from_pretrained(tmpdir).eval()
            with torch.no_grad():
                actual = reloaded(input_ids).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_greedy_generation(self):
        config = self.get_tiny_config()
        model = DualBigMoEForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        out = model.generate(input_ids, max_new_tokens=5, do_sample=False)
        self.assertEqual(out.shape, (1, 9))

    def test_routing_report(self):
        config = self.get_tiny_config()
        model = DualBigMoEForCausalLM(config).eval()
        model.reset_routing_stats()
        input_ids = torch.randint(0, config.vocab_size, (1, 6))
        with torch.no_grad():
            model(input_ids)
        report = model.routing_report()
        self.assertIn("arts_ratio", report)
        self.assertIn("group_shares", report)
        self.assertEqual(len(report["expert_calls"]), config.num_experts)
        self.assertAlmostEqual(sum(report["group_shares"]), 1.0, places=5)
