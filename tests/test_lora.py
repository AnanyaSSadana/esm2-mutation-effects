import unittest

import torch
from torch import nn

from esm2_mutation_effects.lora import (
    LoRALinear,
    count_trainable_parameters,
    inject_lora,
)


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.enable_torch_version = True
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.out_proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.out_proj(self.q_proj(inputs) + self.k_proj(inputs) + self.v_proj(inputs))


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, inputs):
        return self.fc2(torch.relu(self.fc1(self.self_attn(inputs))))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(), ToyLayer()])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class LoRATests(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_output(self):
        torch.manual_seed(7)
        base = nn.Linear(5, 3)
        inputs = torch.randn(4, 5)
        expected = base(inputs).detach()
        layer = LoRALinear(base, rank=2, alpha=2, dropout=0.0)
        self.assertTrue(torch.equal(layer(inputs), expected))

    def test_injection_freezes_base_and_replaces_every_target(self):
        torch.manual_seed(8)
        model = ToyModel().eval()
        inputs = torch.randn(3, 4)
        expected = model(inputs).detach()
        result = inject_lora(model, rank=2, alpha=2, dropout=0.0)

        self.assertEqual(len(result.replaced_modules), 12)
        self.assertTrue(torch.equal(model(inputs), expected))
        self.assertFalse(model.layers[0].self_attn.enable_torch_version)
        # Per layer: four 4->4 projections plus 4->8 and 8->4 feed-forward layers.
        expected_per_layer = 4 * 2 * (4 + 4) + 2 * (4 + 8) + 2 * (8 + 4)
        self.assertEqual(result.trainable_parameters, 2 * expected_per_layer)
        self.assertEqual(result.trainable_parameters, count_trainable_parameters(model))
        self.assertTrue(all(isinstance(model.layers[0].__getattr__(name), LoRALinear) for name in ("fc1", "fc2")))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in model.modules()
                if isinstance(module, LoRALinear)
                for parameter in module.base.parameters()
            )
        )

    def test_adapter_receives_gradients(self):
        torch.manual_seed(9)
        model = ToyModel()
        inject_lora(model, rank=2, alpha=2, dropout=0.0)
        loss = model(torch.randn(3, 4)).square().mean()
        loss.backward()
        gradients = [
            module.lora_b.weight.grad
            for module in model.modules()
            if isinstance(module, LoRALinear)
        ]
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in gradients))

    def test_expected_esm2_t6_adapter_parameter_count(self):
        from esm.model.esm2 import ESM2

        model = ESM2(num_layers=6, embed_dim=320, attention_heads=20)
        result = inject_lora(model, rank=16, alpha=16, dropout=0.05)
        self.assertEqual(len(result.replaced_modules), 36)
        self.assertEqual(result.trainable_parameters, 552_960)

    def test_injection_preserves_fair_esm_forward_output(self):
        from esm.model.esm2 import ESM2

        torch.manual_seed(10)
        model = ESM2(num_layers=1, embed_dim=16, attention_heads=4).eval()
        tokens = torch.tensor([[0, 5, 23, 9, 2]])
        with torch.inference_mode():
            expected = model(tokens, repr_layers=[1], return_contacts=False)["logits"]
        inject_lora(model, rank=2, alpha=2, dropout=0.0)
        with torch.inference_mode():
            actual = model(tokens, repr_layers=[1], return_contacts=False)["logits"]
        self.assertTrue(torch.equal(actual, expected))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(2, 2), rank=0)
        with self.assertRaises(ValueError):
            inject_lora(ToyModel(), target_modules=("missing",))


if __name__ == "__main__":
    unittest.main()
