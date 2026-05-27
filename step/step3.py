"""
Step 3: Compute RadSEM score from tag JSONL and write score JSONL.
"""
import json
import os
import sys
import math
import logging
from collections import deque
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    _radsem_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _radsem_root not in sys.path:
        sys.path.insert(0, _radsem_root)
try:
    from step.step1 import load_existing_names
except ModuleNotFoundError:
    _step_dir = os.path.dirname(os.path.abspath(__file__))
    if _step_dir not in sys.path:
        sys.path.insert(0, _step_dir)
    from step1 import load_existing_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_EPS = 1e-12
_VALID_NORMALITIES = ('abnormal', 'normal')
_BASE_WEIGHTS = {'abnormal': 0.9, 'normal': 0.1}


def _clean_sentence(value):
    return (value or '').strip()


def _pair_weight(pair):
    part_whole_count = 0
    if pair.get('anatomical_relationship') == 'part-whole':
        part_whole_count += 1
    if pair.get('asserted_abnormality_relationship') == 'part-whole':
        part_whole_count += 1
    if pair.get('negated_abnormality_relationship') == 'part-whole':
        part_whole_count += 1

    part_whole_coeff = (1.0 / 3.0) ** part_whole_count

    details_rel = pair.get('details_of_abnormality_relationship')
    if details_rel == 'partial':
        details_coeff = 0.75
    elif details_rel == 'none':
        details_coeff = 0.5
    else:
        details_coeff = 1.0

    return part_whole_coeff * details_coeff


class _Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        fwd = [to, cap, None]
        rev = [fr, 0.0, fwd]
        fwd[2] = rev
        self.graph[fr].append(fwd)
        self.graph[to].append(rev)

    def _bfs(self, s, t):
        level = [-1] * self.n
        level[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for to, cap, _ in self.graph[v]:
                if cap > _EPS and level[to] < 0:
                    level[to] = level[v] + 1
                    if to == t:
                        return level
                    q.append(to)
        return level

    def _dfs(self, v, t, f, level, it):
        if v == t:
            return f
        while it[v] < len(self.graph[v]):
            edge = self.graph[v][it[v]]
            to, cap, rev = edge
            if cap > _EPS and level[v] + 1 == level[to]:
                d = self._dfs(to, t, min(f, cap), level, it)
                if d > _EPS:
                    edge[1] -= d
                    rev[1] += d
                    return d
            it[v] += 1
        return 0.0

    def max_flow(self, s, t):
        flow = 0.0
        while True:
            level = self._bfs(s, t)
            if level[t] < 0:
                break
            it = [0] * self.n
            while True:
                f = self._dfs(s, t, float('inf'), level, it)
                if f <= _EPS:
                    break
                flow += f
        return flow


def _collect_weighted_pairs(pairs):
    weighted_pairs = {'abnormal': [], 'normal': []}
    for idx, pair in enumerate(pairs or []):
        normality = pair.get('normality')
        ref_sentence = _clean_sentence(pair.get('ref_sentence'))
        gen_sentence = _clean_sentence(pair.get('gen_sentence'))
        if normality not in weighted_pairs or not ref_sentence or not gen_sentence:
            continue
        weighted_pairs[normality].append({
            'idx': idx,
            'ref_sentence': ref_sentence,
            'gen_sentence': gen_sentence,
            'weight': _pair_weight(pair),
        })
    return weighted_pairs


def _deduplicate_pair_edges(weighted_pairs):
    by_pair = {}
    for item in weighted_pairs:
        key = (item['ref_sentence'], item['gen_sentence'])
        current = by_pair.get(key)
        if current is None or (item['weight'], -item['idx']) > (current['weight'], -current['idx']):
            by_pair[key] = item
    return list(by_pair.values())


def _allocated_match_count(weighted_pairs):
    weighted_pairs = _deduplicate_pair_edges(weighted_pairs)
    if not weighted_pairs:
        return 0.0

    ref_nodes = {item['ref_sentence']: i for i, item in enumerate({item['ref_sentence']: item for item in weighted_pairs}.values())}
    gen_nodes = {item['gen_sentence']: i for i, item in enumerate({item['gen_sentence']: item for item in weighted_pairs}.values())}

    source = 0
    ref_offset = 1
    gen_offset = ref_offset + len(ref_nodes)
    sink = gen_offset + len(gen_nodes)
    dinic = _Dinic(sink + 1)

    for ref_sentence, idx in ref_nodes.items():
        dinic.add_edge(source, ref_offset + idx, 1.0)
    for gen_sentence, idx in gen_nodes.items():
        dinic.add_edge(gen_offset + idx, sink, 1.0)
    for item in weighted_pairs:
        dinic.add_edge(
            ref_offset + ref_nodes[item['ref_sentence']],
            gen_offset + gen_nodes[item['gen_sentence']],
            item['weight']
        )

    return dinic.max_flow(source, sink)


def _unmatched_counts(unmatched_sentences):
    unmatched_ref = {'abnormal': 0, 'normal': 0}
    unmatched_gen = {'abnormal': 0, 'normal': 0}
    for unmatched in unmatched_sentences or []:
        normality = unmatched.get('normality')
        sentence_from = unmatched.get('sentence_is_from')
        if normality not in unmatched_ref:
            continue
        if sentence_from == 'Ref':
            unmatched_ref[normality] += 1
        elif sentence_from == 'Gen':
            unmatched_gen[normality] += 1
    return unmatched_ref, unmatched_gen


def _unique_pair_weights(weighted_pairs):
    seen = set()
    weights = []
    for item in weighted_pairs:
        key = (item['ref_sentence'], item['gen_sentence'])
        if key in seen:
            continue
        seen.add(key)
        weights.append(item['weight'])
    return weights


def _class_score(matched_count, unmatched_ref_count, unmatched_gen_count, pair_weights):
    denominator = 2 * matched_count + unmatched_ref_count + unmatched_gen_count
    score = (2 * matched_count / denominator) if denominator > 0 else 0.0

    if unmatched_ref_count + unmatched_gen_count == 0:
        score = 1.0
        if pair_weights and any(w < 1.0 - _EPS for w in pair_weights):
            q = sum(pair_weights) / len(pair_weights)
            lam = 0.25 / math.sqrt(len(pair_weights))
            score = 1.0 - lam * (1.0 - q)

    return max(0.0, min(1.0, score))


def calculate_score(tag_result):
    """
    Compute RadSEM score from tag_result (pairs + unmatched_sentences).
    """
    if not tag_result or not isinstance(tag_result, dict):
        return 0.0

    pairs = tag_result.get('pairs', []) or []
    unmatched_sentences = tag_result.get('unmatched_sentences', []) or []

    weighted_pairs = _collect_weighted_pairs(pairs)
    unmatched_ref, unmatched_gen = _unmatched_counts(unmatched_sentences)

    matched_count = {
        normality: _allocated_match_count(weighted_pairs[normality])
        for normality in _VALID_NORMALITIES
    }
    pair_weights = {
        normality: _unique_pair_weights(weighted_pairs[normality])
        for normality in _VALID_NORMALITIES
    }
    exists = {
        normality: bool(weighted_pairs[normality] or unmatched_ref[normality] or unmatched_gen[normality])
        for normality in _VALID_NORMALITIES
    }
    class_scores = {
        normality: _class_score(
            matched_count[normality],
            unmatched_ref[normality],
            unmatched_gen[normality],
            pair_weights[normality]
        )
        for normality in _VALID_NORMALITIES
    }

    present_weights = [(_BASE_WEIGHTS[n], class_scores[n]) for n in _VALID_NORMALITIES if exists[n]]
    if not present_weights:
        return 0.0

    total_weight = sum(weight for weight, _ in present_weights)
    score = sum(weight * value for weight, value in present_weights) / total_weight
    return max(0.0, min(1.0, float(score)))


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
    mean_score = np.mean(score_list)
    logging.info(f"Step 3 done. Total: {total}, Output: {score_file}, mean score: {mean_score}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run_step3(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python step3.py <tag_file> <score_file>")
