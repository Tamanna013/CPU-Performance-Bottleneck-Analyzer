import sys
import os
import unittest
import pandas as pd

# Ensure the src directory is on the import path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from features import extract_features

class TestFeatureExtraction(unittest.TestCase):
    def test_simple_sequence(self):
        df = pd.DataFrame({"address": [0x0, 0x40, 0x80, 0x0]})
        df = extract_features(df)
        # Block numbers should be 0,1,2,0
        self.assertListEqual(df["block_number"].tolist(), [0, 1, 2, 0])
        # Stride blocks: first is 0, then 1, then 1, then -2 (0-2)
        self.assertListEqual(df["stride_blocks"].tolist(), [0, 1, 1, -2])
        # Reuse distance for the last access (block 0) should be 2 distinct blocks between previous 0 and now
        self.assertEqual(df["reuse_distance"].iloc[3], 2)
        # recent_unique_blocks window=8; after first three distinct blocks, count should be 3 before the repeat
        self.assertEqual(df["recent_unique_blocks"].iloc[2], 3)
        # recent_same_block for the last entry (block 0) should be 1 because 0 appeared in recent window
        self.assertEqual(df["recent_same_block"].iloc[3], 1)

if __name__ == "__main__":
    unittest.main()
