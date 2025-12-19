import unittest
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import shutil

class TestAnalysisPhase(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'tests/temp_analysis_results'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_csv_generation_and_loading(self):
        """Test that data can be saved to CSV and reloaded correctly."""
        data = [
            [1, 0.5, -100, 85.0, 0.4, 0.3, 0.2],
            [2, 0.6, -90, 82.0, 0.35, 0.25, 0.15]
        ]
        columns = ["Arch Index", "Lipschitz", "SynFlow", "Test Acc", "Cert Acc (2/255)", "Cert Acc (4/255)", "Cert Acc (8/255)"]
        df = pd.DataFrame(data, columns=columns)
        
        csv_path = os.path.join(self.test_dir, 'test_results.csv')
        df.to_csv(csv_path, index=False)
        
        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_correlation_logic(self):
        """Test Pearson correlation calculation on synthetic data."""
        # Perfect negative correlation between log(Lip) and Cert Acc
        # Lip = e^x, Acc = -x
        
        lip_values = np.exp(np.array([1, 2, 3, 4, 5]))
        cert_accs = np.array([5, 4, 3, 2, 1])
        
        # Log(Lip) = [1, 2, 3, 4, 5]
        log_lips = np.log(lip_values)
        
        r, p = pearsonr(log_lips, cert_accs)
        
        self.assertAlmostEqual(r, -1.0)

if __name__ == '__main__':
    unittest.main()
