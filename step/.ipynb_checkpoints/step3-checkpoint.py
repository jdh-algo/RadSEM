"""
Step 3: Compute RadSEM score from tag JSONL and write score JSONL.
"""
import json
import os
import sys
import logging
from tqdm import tqdm
try:
    import numpy as np
    _has_numpy = True
except ImportError:
    _has_numpy = False

# Support both package import and direct script run
if __name__ == "__main__":
    _radsem_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _radsem_root not in sys.path:
        sys.path.insert(0, _radsem_root)
from step.step1 import load_existing_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_score(tag_result):
    """
    Compute RadSEM score from tag_result (pairs + unmatched_sentences).

    Args:
        tag_result: dict with 'pairs' and 'unmatched_sentences'

    Returns:
        score: float in [0, 1]
    """
    if not tag_result or not isinstance(tag_result, dict):
        return 0.0
    pairs = tag_result.get('pairs', [])
    unmatched_sentences = tag_result.get('unmatched_sentences', [])
    matched_abnormal_count = 0.0
    matched_normal_count = 0.0
    unmatched_ref_abnormal = 0
    unmatched_ref_normal = 0
    unmatched_gen_abnormal = 0
    unmatched_gen_normal = 0
    for pair in pairs:
        normality = pair.get('normality', '')
        if not normality:
            continue
        part_whole_count = 0
        if pair.get('anatomical_relationship') == 'part-whole':
            part_whole_count += 1
        if pair.get('asserted_abnormality_relationship') == 'part-whole':
            part_whole_count += 1
        if pair.get('negated_abnormality_relationship') == 'part-whole':
            part_whole_count += 1
        part_whole_coeff = (1.0 / 3.0) ** part_whole_count
        details_coeff = 1.0
        details_rel = pair.get('details_of_abnormality_relationship')
        if details_rel == 'equivalent':
            details_coeff = 1.0
        elif details_rel == 'partial':
            details_coeff = 0.75
        elif details_rel == 'none':
            details_coeff = 0.5
        elif details_rel is None:
            details_coeff = 1.0
        sentence_weight = part_whole_coeff * details_coeff
        if normality == 'abnormal':
            matched_abnormal_count += sentence_weight
        elif normality == 'normal':
            matched_normal_count += sentence_weight
    for unmatched in unmatched_sentences:
        sentence_from = unmatched.get('sentence_is_from', '')
        normality = unmatched.get('normality', '')
        if normality == 'abnormal':
            if sentence_from == 'Ref':
                unmatched_ref_abnormal += 1
            elif sentence_from == 'Gen':
                unmatched_gen_abnormal += 1
        elif normality == 'normal':
            if sentence_from == 'Ref':
                unmatched_ref_normal += 1
            elif sentence_from == 'Gen':
                unmatched_gen_normal += 1
    abnormal_denominator = 2 * matched_abnormal_count + unmatched_ref_abnormal + unmatched_gen_abnormal
    abnormal_f1 = (2 * matched_abnormal_count / abnormal_denominator) if abnormal_denominator > 0 else 0.0
    normal_denominator = 2 * matched_normal_count + unmatched_ref_normal + unmatched_gen_normal
    normal_f1 = (2 * matched_normal_count / normal_denominator) if normal_denominator > 0 else 0.0
    if unmatched_ref_abnormal + unmatched_gen_abnormal == 0:
        abnormal_f1 = 1.0
    if unmatched_ref_normal + unmatched_gen_normal == 0:
        normal_f1 = 1.0
    score = 0.9 * abnormal_f1 + 0.1 * normal_f1
    return score


def run_step3(tag_file, score_file):
    """
    Run step 3: read tag_file, compute score for each record, write score_file.
    Skips names already present in score_file.
    """
    existing_names = load_existing_names(score_file)
    tag_records = []
    try:
        with open(tag_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if existing_names:
                        record_name = record.get('name')
                        if record_name in existing_names:
                            logging.info(f"Line {line_num} name={record_name} already exists, skip")
                            continue
                    tag_records.append((line_num, record))
                except json.JSONDecodeError as e:
                    logging.error(f"Line {line_num} JSON parse error: {e}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {tag_file}")
        return
    total = len(tag_records)
    if total == 0:
        logging.info("Step 3: No records to process, all already exist.")
        return
    file_mode = 'a' if existing_names else 'w'
    score_list = []
    with open(score_file, file_mode, encoding='utf-8') as out_f:
        for line_num, record in tqdm(tag_records, desc="Step 3 progress"):
            name = record.get('name', '')
            Examined_Area = record.get('Examined_Area', '')
            Examined_Type = record.get('Examined_Type', '')
            findings = record.get('findings', {})
            score = calculate_score(findings)
            score_list.append(score)
            output_record = {
                'name': name,
                'Examined_Area': Examined_Area,
                'Examined_Type': Examined_Type,
                'score': score
            }
            json_line = json.dumps(output_record, ensure_ascii=False)
            out_f.write(json_line + '\n')
            out_f.flush()
    mean_score = np.mean(score_list) if _has_numpy else (sum(score_list) / len(score_list) if score_list else 0.0)
    logging.info(f"Step 3 done. Total: {total}, Output: {score_file}, mean score: {mean_score}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run_step3(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python step3.py <tag_file> <score_file>")
