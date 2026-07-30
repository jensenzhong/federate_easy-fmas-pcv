# Data Audit Report

## Canonical Dataset

- Path: `Data/all_Data/Client_Data_Split_Cleaned.csv`
- Status: **PASS**
- Rows: 688
- Null cells: 0
- Duplicate rows: 0
- Client distribution: {'Client 1': 233, 'Client 2': 235, 'Client 3': 220}
- Target range: 126,358.00 to 11,481,219.70

## Processed Dataset

- Path: `Data/processed/Client_Data_Split_Cleaned_EN.csv`
- Rows: 688
- Columns: BIDS, CPPR, ContDays, LEN, LANE, WeatherDelayDays, CPI, ContSpend, PLR, PPI, ContAmnt, Client
- Null cells: 0
- Duplicate rows: 0

## Legacy Company Files

- `Company_A_train.csv`: 90 rows, ContAmnt 126,358.00-999,999.99; Historical amount-stratified training file; not a current client split.
- `Company_B_train.csv`: 338 rows, ContAmnt 1,010,622.00-4,994,247.32; Historical amount-stratified training file; not a current client split.
- `Company_C_train.csv`: 113 rows, ContAmnt 5,046,000.00-20,277,220.00; Historical amount-stratified training file; not a current client split.

## Interpretation

The current experiments should use the 688-row client-labelled dataset. The old Company_A/B/C files are retained only as historical amount strata.
