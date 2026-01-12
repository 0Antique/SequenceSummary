"""Batch version of RunAll.py.

Runs the same mining pipeline as RunAll.py, but for every CSV file in a directory
and writes all generated JSON outputs into the visualization assets folder.
"""

import os
import sys
import json
import csv
import argparse
import time
from datetime import timedelta
from glob import glob

try:
    from memory_profiler import memory_usage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    memory_usage = None

from utils.args_parser import (
    get_common_parser,
    add_coreflow_args,
    add_sequencesynopsis_args,
)
from utils.data_loader import load_event_store, generate_sequences, ensure_output_directory

from core.Node import TreeNode
from datamodel.Sequence import Sequence
from core.Graph import Graph
from coreflow.CoreFlowMiner import CoreFlowMiner
from sententree.SentenTreeMiner import SentenTreeMiner
from sequencesynopsis.SequenceSynopsisMinerWithWeightedLSH import SequenceSynopsisMiner
from sequencesynopsis.SequenceSynopsisMiner import SequenceSynopsisMiner as ssmv


DEFAULT_INPUT_DIR = "./events2"
DEFAULT_OUTPUT_DIR = "./visualization/app/public/assets"


def _iter_csv_files(input_dir: str, pattern: str):
    paths = glob(os.path.join(input_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    return sorted(paths)


def _ensure_time_memory_header(csv_path: str) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as time_file:
        writer = csv.writer(time_file)
        writer.writerow(["Dataset", "Support", "Tool", "Time", "Memory"])


def _run_with_memory_profile(func, func_args: list):
    """Run a function and (optionally) collect peak memory usage.

    Returns (mem, retval) where mem is a float (MiB) when memory_profiler is
    available, otherwise the string "NA".
    """
    if memory_usage is None:
        return "NA", func(*func_args)
    mem, retval = memory_usage(
        proc=[func, func_args],
        include_children=True,
        max_usage=True,
        retval=True,
    )
    return mem, retval


def _run_one_file(args: argparse.Namespace, csv_file: str, time_writer: csv.writer) -> None:
    args.file = csv_file
    basename = os.path.splitext(os.path.basename(args.file))[0]

    event_store = load_event_store(args)
    seq_list = generate_sequences(event_store, args)
    Sequence.seqListtotsv(seq_list, args.attr)

    min_sup_param = 0.05
    while min_sup_param <= 0.3:
        cfm = CoreFlowMiner(
            args.attr, minSup=min_sup_param * len(seq_list), maxSup=len(seq_list)
        )
        start = time.time()
        mem, output = _run_with_memory_profile(cfm.runCoreFlowMiner, [seq_list])
        end = time.time()
        root = output[0]

        time_writer.writerow(
            [basename, f"{min_sup_param:.2f}", "Coreflow", timedelta(seconds=end - start), mem]
        )

        coreflow_json = json.dumps(
            root, ensure_ascii=False, default=TreeNode.jsonSerializeDump, indent=1
        )
        with open(
            os.path.join(
                args.output, f"{basename}+coreflow_msp{min_sup_param:.2f}.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(coreflow_json)

        stm = SentenTreeMiner(
            args.attr, minSup=min_sup_param * len(seq_list), maxSup=len(seq_list)
        )
        start = time.time()
        mem, graph = _run_with_memory_profile(stm.runSentenTreeMiner, [seq_list])
        end = time.time()
        time_writer.writerow(
            [
                basename,
                f"{min_sup_param:.2f}",
                "Sententree",
                timedelta(seconds=end - start),
                mem,
            ]
        )

        sententree_json = json.dumps(
            graph, ensure_ascii=False, default=Graph.jsonSerializeDump, indent=1
        )
        with open(
            os.path.join(
                args.output, f"{basename}+sententree_msp{min_sup_param:.2f}.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(sententree_json)

        ssm = SequenceSynopsisMiner(
            args.attr, event_store, alpha=min_sup_param, lambdaVal=1 - min_sup_param
        )
        start = time.time()
        mem, output = _run_with_memory_profile(ssm.minDL, [seq_list])
        end = time.time()
        time_writer.writerow(
            [
                basename,
                f"{min_sup_param:.2f}",
                "SyquenceSynopsis",
                timedelta(seconds=end - start),
                mem,
            ]
        )

        grph = output[1]
        seqsynopsis_json = json.dumps(
            grph, ensure_ascii=False, default=Graph.jsonSerializeDump, indent=1
        )
        with open(
            os.path.join(
                args.output, f"{basename}+seqsynopsis_alpha{min_sup_param:.2f}.json"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(seqsynopsis_json)

        ssmvanilla = ssmv(
            args.attr, event_store, alpha=min_sup_param, lambdaVal=1 - min_sup_param
        )
        start = time.time()
        mem, _ = _run_with_memory_profile(ssmvanilla.minDL, [seq_list])
        end = time.time()
        time_writer.writerow(
            [
                basename,
                f"{min_sup_param:.2f}",
                "SyquenceSynopsisvanilla",
                timedelta(seconds=end - start),
                mem,
            ]
        )

        min_sup_param += 0.05


def main() -> None:
    parser = get_common_parser()
    parser = add_coreflow_args(parser)
    parser = add_sequencesynopsis_args(parser)

    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory that contains CSV files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help='Glob pattern for input files (default: "*.csv")',
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Process only the first N files (0 means all).",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately when a file fails.",
    )

    args = parser.parse_args()

    # Force outputs into visualization assets as requested.
    args.output = DEFAULT_OUTPUT_DIR
    ensure_output_directory(args)

    csv_files = _iter_csv_files(args.input_dir, args.pattern)
    if args.max_files and args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {args.input_dir!r} with pattern {args.pattern!r}"
        )

    time_csv_path = "TimeMemoryAnalysis.csv"
    _ensure_time_memory_header(time_csv_path)

    failures = []
    with open(time_csv_path, "a", newline="", encoding="utf-8") as time_file:
        time_writer = csv.writer(time_file)
        for idx, csv_file in enumerate(csv_files, start=1):
            print(f"[{idx}/{len(csv_files)}] Processing: {csv_file}")
            try:
                _run_one_file(args, csv_file, time_writer)
                time_file.flush()
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                failures.append((csv_file, msg))
                print(f"ERROR processing {csv_file}: {msg}", file=sys.stderr)
                if args.fail_fast:
                    raise

    if failures:
        print("\nSome files failed:", file=sys.stderr)
        for csv_file, msg in failures:
            print(f"- {csv_file}: {msg}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
