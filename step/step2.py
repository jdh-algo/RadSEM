"""
Step 2: Sentence matching. Takes two rewritten-report JSONL files (gen and gt),
aligns by name, and produces tag JSONL (pairs + unmatched_sentences).
Both input files must contain rewritten_report (output of step1).
"""
import json
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Support both package import and direct script run
if __name__ == "__main__":
    _radsem_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _radsem_root not in sys.path:
        sys.path.insert(0, _radsem_root)
from step.step1 import call_api, load_existing_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_step2_tag(gen_record, gt_record):
    """
    Step 2: Process one gen record and one gt record (both with rewritten_report), produce tag result.
    Returns: (volumename, Examined_Area, Examined_Type, tag_result) or None.
    """
    try:
        volumename = gen_record.get('name', '')
        Examined_Area = gen_record.get('Examined_Area', '')
        Examined_Type = gen_record.get('Examined_Type', '')
        gen_rewritten_report = gen_record.get('rewritten_report', '')
        gt_rewritten_report = gt_record.get('rewritten_report', '')
        if not gen_rewritten_report or not gt_rewritten_report:
            logging.warning(f"Missing rewritten_report: {volumename}")
            return None
        logging.info(f"Processing volumename: {volumename}, tag generation")
        prompt = f'''
        You are given two radiology reports:

- Ref: {gt_rewritten_report}
- Gen: {gen_rewritten_report}

Task:
1) Identify sentence pairs (Ref ↔ Gen) that have substantially the same clinical meaning. When matching two sentences, you should consider whether the entity/object being described and the situation being described are consistent—or at least stand in an inclusion (containment) relationship. Many-to-many alignment is allowed: a sentence may appear in multiple pairs.
   - Pair sentences when the core clinical statement is the same, even if the granularity differs (e.g., broader vs more specific anatomy/abnormality). Use the relationship labels to capture the difference.

2) Also output sentences from either report that do NOT have any meaningfully similar match in the other report ("unmatched" sentences).

IMPORTANT RULE: Anti-contradiction constraint
Ensure both statements from Gen and Ref could be true at the same time.
If one sentence affirms a finding while the other negates it (e.g., "present/seen" vs "no/absent/not seen"), or they state opposite directions (e.g., increased vs decreased; patent vs obstructed; normal vs abnormal), they are contradictions and MUST NOT be paired; place them in unmatched.

For each paired Ref–Gen sentence, assign these labels:

- normality: "normal" if the paired sentences describe only normal findings; otherwise "abnormal".
- anatomical_relationship: "equivalent" if the anatomical sites are the same; "part-whole" if one site is contained within the other.
- asserted_abnormality_relationship: compare affirmed (present) abnormality concepts; use "equivalent" or "part-whole". If no affirmed abnormality is stated in either sentence, output null.
- negated_abnormality_relationship: compare negated (absent) abnormality concepts; use "equivalent" or "part-whole". If no negated abnormality is stated in either sentence, output null.
"details_of_abnormality":
- Abnormality details are any descriptors that refine an affirmed abnormal finding, such as:
  measurements (size/number), severity (mild/moderate/severe), morphology/appearance (shape/margins/density/signal/enhancement), temporal descriptors (acute/chronic), and diagnostic impression/inference (e.g., "consistent with", "suggesting").
- Exclude:
  (1) anatomical localization words (laterality, organ/region/segment names), and
  (2) the core abnormality concept word(s) themselves (e.g., "nodule", "fracture", "hemorrhage").
- If uncertainty/hedging is present (possible/probable/likely/suspected/cannot exclude/etc.), count the uncertainty cue(s) as part of the details.
- If a sentence contains no such details, treat it as having "no details".

Details-of-abnormality comparison rules:
- Compare abnormality details between Ref and Gen by meaning (not exact string match).
- Output "equivalent" if the meaning coverage is essentially the same.
- Output "partial" if there is meaningful overlap but at least one meaningful detail is missing, extra, or contradictory.
- Output "none" if there is no meaningful overlap, or if one side has no details while the other has details.
- If normality = "normal", set details_of_abnormality_relationship = null.

Output:
Return ONLY one valid JSON object with exactly these two top-level fields: "pairs" and "unmatched_sentences".

Schema:

{{
  "pairs": [
    {{
      "ref_sentence": "<string>",
      "gen_sentence": "<string>",
      "normality": "normal" | "abnormal",
      "anatomical_relationship": "equivalent" | "part-whole",
      "asserted_abnormality_relationship": "equivalent" | "part-whole" ,
      "negated_abnormality_relationship": "equivalent" | "part-whole",
      "details_of_abnormality_relationship": "none" | "partial" | "equivalent"
    }}
  ],
  "unmatched_sentences": [
    {{
      "sentence_is_from": "Ref" | "Gen",
      "sentence": "<string>",
      "normality": "normal" | "abnormal"
    }}
  ]
}}

Return ONLY valid JSON. Do not include any extra text.
'''
        processed_sentence_str = call_api(prompt)
        if not processed_sentence_str:
            logging.warning(f"Tag API call failed: {volumename}")
            return None
        cleaned_str = processed_sentence_str.strip()
        if cleaned_str.startswith('```'):
            lines = cleaned_str.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_str = '\n'.join(lines)
        try:
            json_sentence = json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error {volumename}: {e}")
            logging.error(f"Response: {processed_sentence_str[:200]}...")
            return None
        return volumename, Examined_Area, Examined_Type, json_sentence
    except Exception as e:
        logging.error(f"Error processing record: {e}")
        return None


def run_step2(gen_rewritten_file, gt_rewritten_file, tag_file, max_workers=20, save_batch_size=5):
    """
    Run step 2: load gen_rewritten_file and gt_rewritten_file (both with rewritten_report),
    match by name, produce tag_file. Skips names already present in tag_file.
    """
    existing_names = load_existing_names(tag_file)
    gen_rewritten_data = {}
    try:
        with open(gen_rewritten_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    name = data.get('name')
                    if name:
                        gen_rewritten_data[name] = data
    except FileNotFoundError:
        logging.error(f"File not found: {gen_rewritten_file}")
        return
    gt_rewritten_data = {}
    try:
        with open(gt_rewritten_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    name = data.get('name')
                    if name:
                        gt_rewritten_data[name] = data
    except FileNotFoundError:
        logging.error(f"File not found: {gt_rewritten_file}")
        return
    matched_names = set(gen_rewritten_data.keys()) & set(gt_rewritten_data.keys())
    logging.info(f"Found {len(matched_names)} matched names")
    records = []
    for name in sorted(matched_names):
        if existing_names and name in existing_names:
            logging.info(f"name={name} already exists, skip")
            continue
        records.append((name, gen_rewritten_data[name], gt_rewritten_data[name]))
    total = len(records)
    if total == 0:
        logging.info("Step 2: No records to process, all already exist.")
        return
    file_lock = Lock()
    success_count = 0
    failed_count = 0
    results_dict = {}
    next_write_index = 0
    file_mode = 'a' if existing_names else 'w'
    with open(tag_file, file_mode, encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for idx, (name, gen_record, gt_record) in enumerate(records):
                future = executor.submit(process_step2_tag, gen_record, gt_record)
                future_to_index[future] = (idx, name, gen_record, gt_record)
            with tqdm(total=total, desc="Step 2 progress") as pbar:
                for future in as_completed(future_to_index):
                    idx, name, gen_record, gt_record = future_to_index[future]
                    try:
                        result = future.result()
                        results_dict[idx] = (name, result)
                        if result is not None:
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logging.error(f"name={name} failed: {e}")
                        results_dict[idx] = (name, None)
                        failed_count += 1
                    while next_write_index in results_dict:
                        name, result = results_dict[next_write_index]
                        if result is not None:
                            volumename, Examined_Area, Examined_Type, tag_result = result
                            output_record = {
                                'name': volumename,
                                'Examined_Area': Examined_Area,
                                'Examined_Type': Examined_Type,
                                'findings': tag_result
                            }
                            json_line = json.dumps(output_record, ensure_ascii=False)
                            with file_lock:
                                out_f.write(json_line + '\n')
                                out_f.flush()
                        del results_dict[next_write_index]
                        next_write_index += 1
                        if (next_write_index - 1) % save_batch_size == 0:
                            with file_lock:
                                out_f.flush()
                    pbar.update(1)
                while next_write_index < total:
                    if next_write_index in results_dict:
                        name, result = results_dict[next_write_index]
                        if result is not None:
                            volumename, Examined_Area, Examined_Type, tag_result = result
                            output_record = {
                                'name': volumename,
                                'Examined_Area': Examined_Area,
                                'Examined_Type': Examined_Type,
                                'findings': tag_result
                            }
                            json_line = json.dumps(output_record, ensure_ascii=False)
                            with file_lock:
                                out_f.write(json_line + '\n')
                        del results_dict[next_write_index]
                    next_write_index += 1
                with file_lock:
                    out_f.flush()
    logging.info(f"Step 2 done. Total: {total}, Success: {success_count}, Failed: {failed_count}, Output: {tag_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        run_step2(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python step2.py <gen_rewritten_file> <gt_rewritten_file> <tag_file>")
