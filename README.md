# RadSEM - Radiology Sentence-Level Evaluation Metric

RadSEM is a semantic evaluation metric for radiology reports that breaks down reports into atomic sentences, aligns them between generated and reference reports, and computes detailed scores based on anatomical and abnormality relationships.

## Overview

RadSEM evaluates radiology reports through three main steps:
1. **Step 1 (Report processing)**: Converts reports into atomic sentences following strict rules
2. **Step 2 (Sentence matching)**: Aligns sentences between generated and reference reports with detailed relationship labels
3. **Step 3 (Scoring)**: Computes weighted F1 scores for abnormal and normal findings

## Project Structure

```
RadSEM/
├── l1_l5/                # L1–L5 evaluation data and filtered samples
├── step/
│   ├── step1.py          # Report rewriting into atomic sentences
│   ├── step2.py          # Sentence matching and tagging
│   └── step3.py          # Score calculation
├── run_radsem.py         # Main pipeline orchestrator
├── groundtruth.jsonl     # Reference reports
└── model_output.jsonl    # Generated reports to evaluate
```

## Installation

### API Configuration
The scripts use an API for LLM-based processing. Update the API endpoint and key in `step/step1.py`:
```python
url = "http://your/API/base/url"
headers = {
    "Authorization": "YOUR_API_KEY",
    ...
}
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python run_radsem.py
```

This will:
1. Process `model_output.jsonl` through step1 → `model_rewritten_res.jsonl`
2. Process `groundtruth.jsonl` through step1 → `gt_rewritten_res.jsonl`
3. Align and tag both → `tag.jsonl`
4. Compute scores → `score.jsonl`

## Input Format

### Report Files (JSONL)
Each line should be a JSON object with:
```json
{
  "name": "sample_0001",
  "Examined_Area": "CHEST",
  "Examined_Type": "CT",
  "English_Report": "Both lungs are clear..."
}
```
