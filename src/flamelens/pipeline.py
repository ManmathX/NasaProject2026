"""Run every built flow in order: ``python -m flamelens.pipeline``."""

from __future__ import annotations

from flamelens import coverage, ingest, normalize, tasks, train, whatif


def main() -> None:
    print("== Flow 1: ingest")
    for dataset_id, r in ingest.run().items():
        print(f"  {dataset_id:24s} rows_out={r['rows_out']}")
    print("== Flow 2: normalize")
    normalize.run()
    print("== Flow 3: task tables")
    tasks.run()
    print("== Flow 4: train ext_flex")
    train.run()
    print("== Flow 5: coverage")
    coverage.run()
    print("== Flow 7: maps and physics check")
    whatif.precompute_maps()
    for r in whatif.physics_report():
        print(f"  {r['fuel']:10s} {r['diluent']:4s} flagged={r['flagged']} share_rising={r['share_rising']}")


if __name__ == "__main__":
    main()
