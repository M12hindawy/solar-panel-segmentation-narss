# How to Reproduce All Results

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Full Pipeline**:
   ```bash
   python 10_inference/run_pipeline.py
   ```
3. **Verify Statistical Tests**:
   ```bash
   python -c "import json; print(json.load(open('06_statistical_tests/wilcoxon_results.json')))"
   ```
