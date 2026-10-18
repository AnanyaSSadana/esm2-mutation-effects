import unittest

import torch
from esm.data import Alphabet
from torch import nn

from esm2_mutation_effects.adaptation import (
    AMINO_ACID_TO_INDEX,
    MaskedMarginalResidualModel,
    build_masked_position_inputs,
    encode_mutations,
)
from esm2_mutation_effects.data import load_assay
from esm2_mutation_effects.splits import make_splits


class DummyBackbone(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(40, hidden_dim)

    def forward(self, tokens, repr_layers, return_contacts=False):
        del return_contacts
        layer = repr_layers[0]
        return {"representations": {layer: self.embedding(tokens)}}


class AdaptationTests(unittest.TestCase):
    def test_masked_inputs_use_one_based_residue_token_positions(self):
        alphabet = Alphabet.from_architecture("ESM-1b")
        batch = build_masked_position_inputs("ACDE", [1, 4], alphabet)
        self.assertEqual(batch.residue_positions, (1, 4))
        self.assertEqual(batch.token_positions.tolist(), [1, 4])
        self.assertEqual(int(batch.tokens[0, 1]), alphabet.mask_idx)
        self.assertEqual(int(batch.tokens[1, 4]), alphabet.mask_idx)
        self.assertEqual(alphabet.get_tok(int(batch.tokens[0, 2])), "C")

    def test_mutations_map_to_site_and_amino_acid_indices(self):
        encoded = encode_mutations(["A1C", "E4W"], [1, 4])
        self.assertEqual(encoded.site_indices.tolist(), [0, 1])
        self.assertEqual(
            encoded.wild_type_indices.tolist(),
            [AMINO_ACID_TO_INDEX["A"], AMINO_ACID_TO_INDEX["E"]],
        )
        self.assertEqual(
            encoded.alternate_indices.tolist(),
            [AMINO_ACID_TO_INDEX["C"], AMINO_ACID_TO_INDEX["W"]],
        )

    def test_zero_initialized_model_exactly_returns_fixed_scores(self):
        model = MaskedMarginalResidualModel(DummyBackbone(), hidden_dim=4, representation_layer=1)
        fixed = torch.tensor([0.5, -1.25])
        predicted = model(
            tokens=torch.tensor([[0, 2, 3], [0, 4, 3]]),
            token_positions=torch.tensor([1, 1]),
            site_indices=torch.tensor([0, 1]),
            wild_type_indices=torch.tensor([0, 1]),
            alternate_indices=torch.tensor([2, 3]),
            fixed_zero_shot=fixed,
        )
        self.assertTrue(torch.equal(predicted, fixed))

    def test_prediction_uses_alternate_minus_wild_type_residual(self):
        model = MaskedMarginalResidualModel(DummyBackbone(), hidden_dim=4, representation_layer=1)
        with torch.no_grad():
            model.residual_head.bias[0] = 0.75
            model.residual_head.bias[2] = 2.0
        predicted = model(
            tokens=torch.tensor([[0, 2, 3]]),
            token_positions=torch.tensor([1]),
            site_indices=torch.tensor([0]),
            wild_type_indices=torch.tensor([0]),
            alternate_indices=torch.tensor([2]),
            fixed_zero_shot=torch.tensor([0.5]),
        )
        self.assertAlmostEqual(float(predicted[0].detach()), 1.75, places=6)

    def test_residual_head_and_backbone_can_receive_gradients(self):
        model = MaskedMarginalResidualModel(DummyBackbone(), hidden_dim=4, representation_layer=1)
        with torch.no_grad():
            model.residual_head.weight.normal_(std=0.01)
        prediction = model(
            tokens=torch.tensor([[0, 2, 3]]),
            token_positions=torch.tensor([1]),
            site_indices=torch.tensor([0]),
            wild_type_indices=torch.tensor([0]),
            alternate_indices=torch.tensor([2]),
            fixed_zero_shot=torch.tensor([0.5]),
        )
        prediction.square().sum().backward()
        self.assertIsNotNone(model.residual_head.weight.grad)
        self.assertIsNotNone(model.backbone.embedding.weight.grad)

    def test_position_split_subsets_are_disjoint(self):
        splits = make_splits(load_assay(), seed=42)
        position_sets = {
            subset: set(splits.loc[splits["position"] == subset, "residue_position"])
            for subset in ("train", "validation", "test")
        }
        self.assertTrue(position_sets["train"].isdisjoint(position_sets["validation"]))
        self.assertTrue(position_sets["train"].isdisjoint(position_sets["test"]))
        self.assertTrue(position_sets["validation"].isdisjoint(position_sets["test"]))

    def test_invalid_mask_and_mutation_requests_are_rejected(self):
        alphabet = Alphabet.from_architecture("ESM-1b")
        for positions in ([], [0], [5], [2, 2]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                build_masked_position_inputs("ACDE", positions, alphabet)
        with self.assertRaises(ValueError):
            encode_mutations(["A1C"], [2])


if __name__ == "__main__":
    unittest.main()
