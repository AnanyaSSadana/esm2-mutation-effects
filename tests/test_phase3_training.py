import unittest

import numpy as np
import pandas as pd
import torch

from esm2_mutation_effects.phase3_training import PilotMutationData, sample_tie_free_lists


class Phase3TrainingTests(unittest.TestCase):
    def test_tie_safe_lists_choose_distinct_score_values(self):
        data = PilotMutationData(
            frame=pd.DataFrame({"DMS_score": [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]}),
            masked_tokens=torch.empty(0), token_positions=torch.empty(0), site_indices=torch.empty(0),
            wild_type_indices=torch.empty(0), alternate_indices=torch.empty(0), fixed_zero_shot=torch.empty(0),
            score_bins=(np.array([0, 1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])),
        )
        lists = sample_tie_free_lists(data, np.random.default_rng(1), lists_per_update=8)
        self.assertEqual(lists.shape, (8, 5))
        for values in data.frame.iloc[lists.reshape(-1)].DMS_score.to_numpy().reshape(8, 5):
            self.assertEqual(len(set(values)), 5)


if __name__ == "__main__":
    unittest.main()
