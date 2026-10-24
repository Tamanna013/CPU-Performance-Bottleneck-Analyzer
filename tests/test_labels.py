import sys
import os
import unittest
import pandas as pd

# Ensure the src directory is on import path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from cache_labels import label_trace

class TestCacheLabeling(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp'))
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.trace_path = os.path.join(self.tmp_dir, 'test.trace')
        # Simple trace: first miss, then hit, then miss (new block), then hit
        with open(self.trace_path, "w") as f:
            f.write("0x0\n0x0\n0x40\n0x0\n")
        self.out_csv = os.path.join(self.tmp_dir, 'test.csv')

    def tearDown(self):
        try:
            os.remove(self.trace_path)
            os.remove(self.out_csv)
            os.rmdir(self.tmp_dir)
        except Exception:
            pass

    def test_label_output(self):
        label_trace(self.trace_path, self.out_csv)
        df = pd.read_csv(self.out_csv)
        # Expect labels: [1 (miss), 0 (hit), 1 (miss), 0 (hit)]
        self.assertEqual(df["label"].tolist(), [1, 0, 1, 0])

if __name__ == "__main__":
    unittest.main()
