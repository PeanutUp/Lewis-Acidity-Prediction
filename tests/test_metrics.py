import csv
import tempfile
import unittest
from pathlib import Path
from lewis_acidity.metrics import compare, original_table_audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, rows):
        path = self.root / name
        with path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['mol_id', 'y_true', 'y_pred'])
            w.writerows(rows)
        return path

    def test_order_independent_alignment_and_metrics(self):
        a = self.write('a.csv', [('a', 0, 1), ('b', 2, 1)])
        b = self.write('b.csv', [('b', 2, 2), ('a', 0, 0)])
        result = compare([a, b])
        self.assertEqual(result[str(a)]['mae'], 1)
        self.assertEqual(result[str(a)]['r2'], 0)
        self.assertEqual(result[str(b)]['r2'], 1)

    def test_reject_different_test_ids(self):
        a = self.write('a.csv', [('a', 0, 0), ('b', 1, 1)])
        b = self.write('b.csv', [('a', 0, 0), ('c', 1, 1)])
        with self.assertRaisesRegex(ValueError, 'test IDs differ'):
            compare([a, b])

    def test_reject_different_labels(self):
        a = self.write('a.csv', [('a', 0, 0), ('b', 1, 1)])
        b = self.write('b.csv', [('a', 0, 0), ('b', 10, 1)])
        with self.assertRaisesRegex(ValueError, 'target mismatch'):
            compare([a, b])

    def test_reject_duplicate_ids_and_nonfinite(self):
        for rows in [[('a', 0, 0), ('a', 1, 1)], [('a', 0, 'nan'), ('b', 1, 1)]]:
            with self.assertRaises(ValueError):
                compare([self.write('bad.csv', rows)])

    def test_constant_target_has_undefined_r2(self):
        p = self.write('constant.csv', [('a', 1, 1), ('b', 1, 2)])
        self.assertIsNone(compare([p])[str(p)]['r2'])

    def test_original_inconsistency_survives_rounding(self):
        self.assertFalse(original_table_audit()['rounding_intervals_overlap'])


if __name__ == '__main__':
    unittest.main()
