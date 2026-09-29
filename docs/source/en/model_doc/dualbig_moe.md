<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# DualBigMoE

DualBigLittle-MoE is a Qwen3-based architecture where every decoder layer's MLP is replaced by a
dual-big-core + micro-expert FFN (`DualBigMoEMLP`). Tier 1 is the frozen original MLP, Tier 2 is a
fine-tuned clone, and Tier 3 is a pool of LoRA micro-experts streamed from pinned host memory.

The example below demonstrates how to generate text with the [`AutoModelForCausalLM`] class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-org/dualbig-qwen3-0.6b")
model = AutoModelForCausalLM.from_pretrained("your-org/dualbig-qwen3-0.6b", device_map="auto")
inputs = tokenizer("解释一下快速排序", return_tensors="pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0])
```

## Notes

- Micro-experts are selected by the last token only, so a whole prompt shares one expert set.
- Set `expert_pool_location="device"` to keep experts resident on GPU (numerically identical, no PCIe transfer).
- `config.top_k_experts` is deliberately not called `top_k` to avoid colliding with `GenerationConfig`.

## Transformers.DualBigMoEConfig

[[autodoc]] DualBigMoEConfig

## Transformers.DualBigMoEModel

[[autodoc]] DualBigMoEModel
    - forward

## Transformers.DualBigMoEForCausalLM

[[autodoc]] DualBigMoEForCausalLM
    - forward
