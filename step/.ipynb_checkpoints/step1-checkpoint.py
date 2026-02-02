"""
Step 1: Rewrite report into atomic sentences (rewritten_report).
Processes an input JSONL file (e.g. gen_file or gt_file) and writes an output JSONL
with name, Examined_Area, Examined_Type, rewritten_report.
"""
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def call_api(prompt):
    """Call API to process text."""
    try:
        response = requests.post(
            url="http://your/API/base/url",
            headers={
                "Content-Type": "application/json",
                "Authorization": "your API key",
                "Accept": "application/json"
            },
            json={
                "model": "gpt-5",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16392,
                "temperature": 1
            }
        )
        if response:
            gpt_return_list = response.json().get('choices')
            item = gpt_return_list[0]
            gpt = item.get('message').get('content')
            return gpt
        return None
    except Exception as e:
        logging.error(f"API call error: {e}")
        return None


def clean_api_response(response_text):
    """
    Clean API response for JSON parsing.
    1) Remove markdown code block markers.
    2) Remove blank lines.
    3) Escape control characters inside JSON string values.
    """
    if not response_text:
        return None
    import re
    cleaned = response_text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        cleaned = '\n'.join(lines)
    lines = [line for line in cleaned.split('\n') if line.strip()]
    cleaned = '\n'.join(lines)
    result = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(cleaned):
        char = cleaned[i]
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif in_string and ord(char) < 32:
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif char == '\b':
                result.append('\\b')
            elif char == '\f':
                result.append('\\f')
            else:
                result.append(f'\\u{ord(char):04x}')
        else:
            result.append(char)
        i += 1
    cleaned = ''.join(result)
    return cleaned.strip()


def load_existing_names(output_file):
    """
    If output file exists, read all 'name' fields.
    Returns: set of names.
    """
    existing_names = set()
    try:
        if os.path.exists(output_file):
            logging.info(f"Output file exists: {output_file}, reading existing names...")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        name = data.get('name')
                        if name:
                            existing_names.add(name)
                    except json.JSONDecodeError:
                        continue
            logging.info(f"Read {len(existing_names)} existing names.")
        else:
            logging.info(f"Output file does not exist, will create: {output_file}")
    except Exception as e:
        logging.warning(f"Error reading existing file: {e}, will process all records.")
    return existing_names


def process_step1_rewrite(record):
    """
    Step 1: Process one record to produce rewritten_report.
    Expects record to have English_Report (Findings section).
    Returns: (volumename, Examined_Area, Examined_Type, rewritten_report) or None.
    """
    try:
        findings_en = record.get('English_Report', '')
        if not findings_en:
            logging.warning("Missing English_Report field")
            return None
        volumename = record.get('name', '')
        Examined_Area = record.get('Examined_Area', '')
        Examined_Type = record.get('Examined_Type', '')
        logging.info(f"Processing volumename: {volumename}, text splitting")
        prompt = """Task: Rewrite a medical imaging report into atomic sentences under strict constraints.
You MUST use only the Findings section as input. Do not read, reference, or derive anything from Impression or any other section.
Definitions
- "Sentence" = one atomic finding statement.
- Allowed sentence forms ONLY:
  1) <Anatomical location> + <normal finding>
  2) <Anatomical location> + <one abnormal finding>
- "Paired organ" includes left/right sides (e.g., lungs, kidneys, breasts, ovaries, etc.).
- If the report contains multiple MRI sequences, each sentence must belong to EXACTLY ONE sequence.
-If any foreign bodies are identified on imaging—such as tubes/catheters or other medical devices—these should also be considered as an ABNORMAL finding.

Rules (apply in order)
R1. Atomicity of findings
- Each sentence must contain exactly ONE finding.
- A sentence cannot mix normal + abnormal.
- A sentence cannot contain multiple abnormal findings.
=> If violated, split into multiple sentences until every sentence has exactly one finding.

R2. Left/right separation for paired organs
- Do NOT mention both sides in one sentence.
- If a sentence refers to a paired organ without specifying side (e.g., "lungs"), rewrite into TWO sentences:
  - one for the left side
  - one for the right side
- If a sentence explicitly mentions both sides, split into two side-specific sentences.

R3. MRI sequence isolation (only if sequences exist)
- Each sentence must contain information from exactly ONE sequence.
=> If a sentence mixes sequences, split by sequence.

R4. Removal rules
Remove a sentence if:
- it is not in the allowed forms (see Definitions), OR
- it is comparison-to-prior wording (e.g., "compared with prior", "previous study", "interval change"), OR
- it contains non-finding content (history, indication, technique, recommendation, impression headings, measurements not tied to a finding, etc.).

Procedure
1) Split the report into candidate sentences.
2) For each candidate, apply R1–R3 to split as needed.
3) Apply R4 to delete invalid sentences.
4) Output the remaining sentences in the original report order.

Output (STRICT)
Return ONLY valid JSON with exactly one key:
{"rewritten_report":"<sentence1>. <sentence2>. <sentence3>. ..."}
- Each sentence MUST end with a period "." (no exceptions).
- Separate sentences using exactly ONE space after each period.
- Use English.
- Do NOT add any extra keys, commentary, markdown, or explanations.""" + f''' The report is{findings_en}'''
        divided_findings_str = call_api(prompt)
        if not divided_findings_str:
            logging.warning(f"Rewrite API call failed: {volumename}")
            return None
        cleaned_str = divided_findings_str
        try:
            json_res = json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error {volumename}: {e}")
            logging.error(f"Response: {divided_findings_str}")
            return None
        rewritten_report = json_res.get('rewritten_report')
        if not rewritten_report:
            logging.warning(f"rewritten_report not found: {volumename}")
            return None
        return volumename, Examined_Area, Examined_Type, rewritten_report
    except Exception as e:
        logging.error(f"Error processing record: {e}")
        return None


def run_step1(input_file, output_file, max_workers=20, save_batch_size=5):
    """
    Run step 1 on input_file and write results to output_file.
    Skips records whose name already exists in output_file.
    """
    existing_names = load_existing_names(output_file)
    records = []
    try:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for line_num, line in enumerate(in_f, 1):
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
                    records.append((line_num, record))
                except json.JSONDecodeError as e:
                    logging.error(f"Line {line_num} JSON parse error: {e}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        return
    total = len(records)
    if total == 0:
        logging.info("Step 1: No records to process, all already exist.")
        return
    file_lock = Lock()
    success_count = 0
    failed_count = 0
    results_dict = {}
    next_write_index = 0
    file_mode = 'a' if existing_names else 'w'
    with open(output_file, file_mode, encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for idx, (line_num, record) in enumerate(records):
                future = executor.submit(process_step1_rewrite, record)
                future_to_index[future] = (idx, line_num, record)
            with tqdm(total=total, desc="Step 1 progress") as pbar:
                for future in as_completed(future_to_index):
                    idx, line_num, record = future_to_index[future]
                    try:
                        result = future.result()
                        results_dict[idx] = (line_num, result)
                        if result is not None:
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logging.error(f"Line {line_num} failed: {e}")
                        results_dict[idx] = (line_num, None)
                        failed_count += 1
                    while next_write_index in results_dict:
                        line_num, result = results_dict[next_write_index]
                        if result is not None:
                            volumename, Examined_Area, Examined_Type, rewritten_report = result
                            output_record = {
                                'name': volumename,
                                'Examined_Area': Examined_Area,
                                'Examined_Type': Examined_Type,
                                'rewritten_report': rewritten_report
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
                        line_num, result = results_dict[next_write_index]
                        if result is not None:
                            volumename, Examined_Area, Examined_Type, rewritten_report = result
                            output_record = {
                                'name': volumename,
                                'Examined_Area': Examined_Area,
                                'Examined_Type': Examined_Type,
                                'rewritten_report': rewritten_report
                            }
                            json_line = json.dumps(output_record, ensure_ascii=False)
                            with file_lock:
                                out_f.write(json_line + '\n')
                        del results_dict[next_write_index]
                    next_write_index += 1
                with file_lock:
                    out_f.flush()
    logging.info(f"Step 1 done. Total: {total}, Success: {success_count}, Failed: {failed_count}, Output: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        run_step1(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python step1.py <input_file> <output_file>")
