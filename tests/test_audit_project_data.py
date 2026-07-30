import tempfile
import unittest
from pathlib import Path

import pandas as pd


class DataAuditTests(unittest.TestCase):
    def test_audit_current_dataset_reports_expected_quality(self):
        from scripts.audit_project_data import audit_current_dataset

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw_path = base / "Data" / "all_Data" / "Client_Data_Split_Cleaned.csv"
            raw_path.parent.mkdir(parents=True)
            pd.DataFrame([
                {
                    "Dist区域": 1,
                    "BIDS投标数量": 2,
                    "WorkDays工期": 10,
                    "Length长度": 1.5,
                    "Lanes车道数": 2,
                    "ContDays合同日期": 100,
                    "ContAmnt合同金额": 1000000,
                    "CPPR承包商绩效评分": 80,
                    "CPI消费物价指数": 220,
                    "ConstSpndg建设投资支出": 100,
                    "PLR基准贷款利率": 4.5,
                    "PPI (2 year lag)生产物价指数": 180,
                    "ActualAmnt实际结算金额": 1050000,
                    "Client": "Client 1",
                },
                {
                    "Dist区域": 2,
                    "BIDS投标数量": 3,
                    "WorkDays工期": 15,
                    "Length长度": 2.5,
                    "Lanes车道数": 4,
                    "ContDays合同日期": 120,
                    "ContAmnt合同金额": 2000000,
                    "CPPR承包商绩效评分": 85,
                    "CPI消费物价指数": 221,
                    "ConstSpndg建设投资支出": 110,
                    "PLR基准贷款利率": 4.0,
                    "PPI (2 year lag)生产物价指数": 181,
                    "ActualAmnt实际结算金额": 1950000,
                    "Client": "Client 2",
                },
            ]).to_csv(raw_path, index=False)

            audit = audit_current_dataset(base, expected_rows=2)

        self.assertEqual(audit["path"], "Data/all_Data/Client_Data_Split_Cleaned.csv")
        self.assertEqual(audit["rows"], 2)
        self.assertEqual(audit["null_cells"], 0)
        self.assertEqual(audit["duplicate_rows"], 0)
        self.assertEqual(audit["client_counts"], {"Client 1": 1, "Client 2": 1})
        self.assertEqual(audit["status"], "PASS")

    def test_company_files_are_classified_as_legacy_amount_strata(self):
        from scripts.audit_project_data import classify_company_file

        df = pd.DataFrame({
            "ContAmnt": [100000, 500000, 900000],
            "BIDS": [1, 2, 3],
        })

        classification = classify_company_file("Company_A_train.csv", df)

        self.assertEqual(classification["role"], "legacy_amount_stratum")
        self.assertIn("not a current client split", classification["note"])


if __name__ == "__main__":
    unittest.main()
