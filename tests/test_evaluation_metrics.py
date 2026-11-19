"""Unit tests for evaluation metrics in BenchmarkHAR.

Validates that:
1. Macro-F1 is computed over the classes present in the test fold ground truth with zero_division=0.
2. A test fold missing some canonical classes is evaluated fairly on performed activities.
3. Cases where a model predicts a class absent from the test fold ground truth reduce the recall
   of the true class without altering the denominator of present classes.
4. Pooled Macro-F1 concatenates predictions across folds and computes Macro-F1 over pooled present classes.
5. Empty inputs and mismatched prediction lengths raise ValueError.
"""

import unittest
import numpy as np
from sklearn.metrics import f1_score

from src.utils.evaluation import compute_fold_metrics, compute_pooled_metrics


class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        # 12 canonical activities for PAMAP2 protocol
        self.canonical_classes = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

    def test_macro_f1_over_classes_present_in_test_fold(self):
        """Fold Macro-F1 is computed over classes present in the test fold ground truth."""
        y_true = np.array([1, 1, 2, 2, 3, 3])
        y_pred = np.array([1, 1, 2, 2, 3, 3])

        metrics = compute_fold_metrics(y_true, y_pred, canonical_classes=self.canonical_classes)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["present_classes"], [1, 2, 3])
        self.assertEqual(len(metrics["per_class_f1"]), 3)

    def test_test_fold_missing_canonical_classes_not_penalized(self):
        """When a test subject performs only 8 of 12 activities (e.g. PAMAP2 Subject 103),
        primary f1_macro reflects performance on the 8 performed activities (1.0),
        while complementary f1_macro_canonical reflects the full 12-class vocabulary (8/12)."""
        performed_classes = [1, 2, 3, 4, 12, 13, 16, 17]
        y_true = np.repeat(performed_classes, 20)
        y_pred = np.repeat(performed_classes, 20)

        metrics = compute_fold_metrics(y_true, y_pred, canonical_classes=self.canonical_classes)

        # Primary metric evaluates the activities the subject actually performed
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(len(metrics["present_classes"]), 8)
        self.assertEqual(metrics["present_classes"], performed_classes)

        # Complementary metric documents the 12-class fixed denominator
        expected_canonical = 8.0 / 12.0
        self.assertAlmostEqual(metrics["f1_macro_canonical"], expected_canonical, places=5)

    def test_model_predicts_class_absent_from_test_fold(self):
        """When a model hallucinates a class absent from the test fold ground truth,
        the error is penalized via reduced recall on the true class, while the denominator
        remains the set of classes present in the test fold ground truth."""
        # Ground truth only contains classes 1 and 2 (e.g. 10 samples each)
        y_true = np.array([1] * 10 + [2] * 10)

        # For class 1: 10 correct
        # For class 2: 5 correct, 5 incorrectly predicted as class 24 (which is absent from y_true)
        y_pred = np.array([1] * 10 + [2] * 5 + [24] * 5)

        metrics = compute_fold_metrics(y_true, y_pred, canonical_classes=self.canonical_classes)

        # Present classes in ground truth: [1, 2]
        self.assertEqual(metrics["present_classes"], [1, 2])

        # Class 1: TP=10, FP=0, FN=0 -> F1 = 1.0
        # Class 2: TP=5, FP=0, FN=5 -> Precision = 5/5 = 1.0, Recall = 5/10 = 0.5 -> F1 = 2*(1*0.5)/(1+0.5) = 2/3 ≈ 0.6667
        # Macro-F1 over present classes = (1.0 + 2/3) / 2 = 5/6 ≈ 0.8333
        expected_macro_f1 = (1.0 + (2.0 / 3.0)) / 2.0
        self.assertAlmostEqual(metrics["f1_macro"], expected_macro_f1, places=4)
        self.assertAlmostEqual(metrics["per_class_f1"][1], 1.0)
        self.assertAlmostEqual(metrics["per_class_f1"][2], 2.0 / 3.0, places=4)
        self.assertNotIn(24, metrics["per_class_f1"])

    def test_pooled_macro_f1_across_folds(self):
        """Pooled Macro-F1 concatenates predictions across folds and computes Macro-F1
        over the union of classes present in the pooled ground truth."""
        # Fold 1 (Subject A): performed classes 1, 2
        fold1_true = [1, 1, 2, 2]
        fold1_pred = [1, 1, 2, 2]

        # Fold 2 (Subject B): performed classes 3, 4
        fold2_true = [3, 3, 4, 4]
        fold2_pred = [3, 3, 4, 4]

        # Fold 3 (Subject C): performed classes 1, 3, 4, but made 1 error
        fold3_true = [1, 1, 3, 3, 4, 4]
        fold3_pred = [1, 1, 3, 4, 4, 4]  # one 3 predicted as 4

        pooled = compute_pooled_metrics(
            [fold1_true, fold2_true, fold3_true],
            [fold1_pred, fold2_pred, fold3_pred],
            cv_strategy="LOSO",
        )

        self.assertEqual(pooled["pooled_present_classes"], [1, 2, 3, 4])
        self.assertEqual(pooled["n_pooled_samples"], 14)
        self.assertTrue(pooled["is_disjoint_partition"])

        # Ground truth pooled counts:
        # Class 1: 4 samples, 4 correct -> F1 = 1.0
        # Class 2: 2 samples, 2 correct -> F1 = 1.0
        # Class 3: 4 samples, 3 correct, 1 FN -> P=3/3=1.0, R=3/4=0.75 -> F1 = 2*(1*0.75)/(1+0.75) = 1.5/1.75 = 6/7
        # Class 4: 4 samples, 4 correct, 1 FP -> P=4/5=0.8, R=4/4=1.0 -> F1 = 2*(0.8*1)/(0.8+1) = 1.6/1.8 = 8/9
        f1_3 = 6.0 / 7.0
        f1_4 = 8.0 / 9.0
        expected_pooled_f1 = (1.0 + 1.0 + f1_3 + f1_4) / 4.0

        self.assertAlmostEqual(pooled["f1_macro_pooled"], expected_pooled_f1, places=4)
        self.assertAlmostEqual(pooled["accuracy_pooled"], 13.0 / 14.0, places=4)

    def test_pooled_metrics_rejects_empty_inputs(self):
        """Empty fold lists must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_pooled_metrics([], [])

    def test_length_mismatch_raises_error(self):
        """Mismatched y_true and y_pred lengths must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_fold_metrics([1, 2, 3], [1, 2])

    def test_empty_input_raises_error(self):
        """Empty inputs must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_fold_metrics([], [])


if __name__ == "__main__":
    unittest.main()
