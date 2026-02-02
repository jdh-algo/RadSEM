"""
RadSEM pipeline: run step1 (rewrite) on both gen_file and gt_file,
then step2 (tag) on the two rewritten outputs, then step3 (score).
"""
import os
import sys

# Ensure RadSEM directory is on path when run as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from step.step1 import run_step1
from step.step2 import run_step2
from step.step3 import run_step3


def main():
    # File names (under current working directory, typically RadSEM)
    gt_file = "groundtruth.jsonl"
    gen_file = "model_output.jsonl"
    gen_rewritten_file = "model_rewritten_res.jsonl"
    gt_rewritten_file = "gt_rewritten_res.jsonl"
    tag_file = "tag.jsonl"
    score_file = "score.jsonl"

    # Step 1a: process gen_file -> gen_rewritten_file
    print("=" * 60)
    print("Step 1a: Rewrite gen_file -> gen_rewritten_file")
    print("=" * 60)
    run_step1(gen_file, gen_rewritten_file)

    # Step 1b: process gt_file -> gt_rewritten_file (gt also goes through step1)
    print("\n" + "=" * 60)
    print("Step 1b: Rewrite gt_file -> gt_rewritten_file")
    print("=" * 60)
    run_step1(gt_file, gt_rewritten_file)

    # Step 2: tag using gen_rewritten_file and gt_rewritten_file -> tag_file
    print("\n" + "=" * 60)
    print("Step 2: Tag (gen_rewritten + gt_rewritten) -> tag_file")
    print("=" * 60)
    run_step2(gen_rewritten_file, gt_rewritten_file, tag_file)

    # Step 3: compute score from tag_file -> score_file
    print("\n" + "=" * 60)
    print("Step 3: Score tag_file -> score_file")
    print("=" * 60)
    run_step3(tag_file, score_file)

    print("\nRadSEM pipeline finished.")


if __name__ == "__main__":
    main()
