import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "plot_sample_efficiency.py"
SPEC = importlib.util.spec_from_file_location("plot_sample_efficiency", SCRIPT_PATH)
plot_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_module)


class PlotSampleEfficiencyTests(unittest.TestCase):
    def make_summary(self):
        rows = []
        for fraction, labels in ((0.1, 40), (1.0, 400)):
            for model, mean in (
                ("mutation_features", 0.2),
                ("zero_shot_esm2", 0.4),
                ("frozen_esm2", 0.5),
            ):
                rows.append(
                    {
                        "train_fraction": fraction,
                        "model": model,
                        "n_seeds": 5,
                        "mean_labeled_train": 0 if model == "zero_shot_esm2" else labels,
                        "mean_test_spearman": mean,
                        "std_test_spearman": 0.1,
                    }
                )
        return pd.DataFrame(rows)

    def test_load_and_plot_valid_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            summary_path = directory / "summary.csv"
            png_path = directory / "curve.png"
            pdf_path = directory / "curve.pdf"
            self.make_summary().to_csv(summary_path, index=False)

            summary = plot_module.load_summary(summary_path)
            plot_module.plot_summary(summary, png_path, pdf_path, dpi=72)

            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)

    def test_rejects_nonconstant_zero_shot_reference(self):
        summary = self.make_summary()
        summary.loc[
            (summary["model"] == "zero_shot_esm2")
            & (summary["train_fraction"] == 1.0),
            "mean_test_spearman",
        ] = 0.3
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with self.assertRaises(ValueError):
                plot_module.plot_summary(
                    summary,
                    directory / "curve.png",
                    directory / "curve.pdf",
                    dpi=72,
                )


if __name__ == "__main__":
    unittest.main()
