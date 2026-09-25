from __future__ import annotations

import argparse

from baseline_library import run_baseline
from from_scratch import evaluate, load_pairs, train_corrector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-distance", type=int, default=2)
    args = parser.parse_args()
    pairs = load_pairs(args.csv, args.limit)
    corrector = train_corrector(pairs, max_distance=args.max_distance)
    scratch_accuracy, scratch_predictions = evaluate(pairs, corrector)
    library_accuracy, library_predictions, library_time = run_baseline(pairs)
    print("metric                         from_scratch       pyspellchecker")
    print("-" * 70)
    print(f"pairs                         {len(pairs):>16} {len(pairs):>16}")
    print(f"vocabulary_size               {len(corrector.language_model.vocabulary):>16} {'built-in':>16}")
    print(f"exact_accuracy                {scratch_accuracy:>16.4f} {library_accuracy:>16.4f}")
    print(f"runtime_seconds               {'see scratch run':>16} {library_time:>16.4f}")
    print("sample_predictions:")
    for index, ((observed, target), scratch, library) in enumerate(zip(pairs, scratch_predictions, library_predictions)):
        if index >= 20:
            break
        print(f"{observed} -> scratch={scratch} library={library} expected={target}")


if __name__ == "__main__":
    main()
