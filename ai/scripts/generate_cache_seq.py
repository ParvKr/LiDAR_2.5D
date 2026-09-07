import os
import shutil
import time
from pathlib import Path
import sys
import argparse

# ------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ai"))

from data.datasets.bev_dataset import BEVDataset

def main():
    parser = argparse.ArgumentParser(description="Generate cache and backup to a secondary drive.")
    parser.add_argument("--seq", type=str, default="00")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/drive/MyDrive/semantickitti/dataset"))
    parser.add_argument("--local-cache-root", type=Path, default=Path("/content/bev_cache"))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    SEQ = args.seq
    DATASET_ROOT = args.dataset_root
    SEQUENCE_DIR = DATASET_ROOT / "sequences" / SEQ

    CONFIG_NAME = (
        "res_0.2_x_-50.0_50.0"
        "_y_-50.0_50.0"
        "_z_-10.0_35.0"
    )

    DRIVE_CACHE = (
        SEQUENCE_DIR
        / ".bev_cache"
        / CONFIG_NAME
    )

    LOCAL_CACHE = (
        args.local_cache_root
        / SEQ
        / CONFIG_NAME
    )

    BATCH_SIZE = args.batch_size

    # ------------------------------------------------------------
    # CACHE LOCATION
    # ------------------------------------------------------------
    # Force the BEVDataset to write its cache here initially
    os.environ["BEV_CACHE_DIR"] = str(args.local_cache_root.parent)

    # ------------------------------------------------------------
    # DIRECTORIES
    # ------------------------------------------------------------
    DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------
    try:
        dataset = BEVDataset(SEQUENCE_DIR, use_cache=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total = len(dataset)

    print("=" * 70)
    print(f"SEQUENCE {SEQ} CACHE GENERATION")
    print("=" * 70)
    print(f"Total frames : {total}")
    print(f"Drive cache  : {DRIVE_CACHE}")
    print(f"Local cache  : {LOCAL_CACHE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print()

    # ------------------------------------------------------------
    # FIND EXISTING DRIVE FILES
    # ------------------------------------------------------------
    existing = {p.name for p in DRIVE_CACHE.glob("frame_*.pt")}

    print(f"Already cached on Drive : {len(existing)}")
    print(f"Remaining                : {total - len(existing)}\n")

    # ------------------------------------------------------------
    # GENERATE IN BATCHES
    # ------------------------------------------------------------
    remaining_indices = [i for i in range(total) if f"frame_{i:06d}.pt" not in existing]
    overall_start = time.time()
    generated_total = 0

    for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
        batch_indices = remaining_indices[batch_start:batch_start + BATCH_SIZE]
        batch_number = (batch_start // BATCH_SIZE) + 1
        batch_total = len(batch_indices)

        print(f"\nBatch {batch_number} | {batch_total} frames")

        # --------------------------------------------------------
        # Generate entire batch locally
        # --------------------------------------------------------
        batch_start_time = time.time()

        for index in batch_indices:
            filename = f"frame_{index:06d}.pt"
            local_file = LOCAL_CACHE / filename

            # In case a previous interrupted run left it locally
            if local_file.exists():
                continue

            # This triggers the precomputation and local save
            _ = dataset[index]

        generation_time = (time.time() - batch_start_time)
        local_files = list(LOCAL_CACHE.glob("frame_*.pt"))

        print(f"Local cache files ready: {len(local_files)}")
        print(f"Generation time: {generation_time:.1f}s")

        # --------------------------------------------------------
        # Copy batch to Drive
        # --------------------------------------------------------
        print("Copying batch to Google Drive...")
        copy_start = time.time()
        copied = 0

        for local_file in local_files:
            drive_file = DRIVE_CACHE / local_file.name
            if drive_file.exists():
                local_file.unlink()
                continue

            shutil.copy2(local_file, drive_file)
            copied += 1

        copy_time = time.time() - copy_start
        print(f"Copied {copied} files in {copy_time:.1f}s")

        # --------------------------------------------------------
        # VERIFY
        # --------------------------------------------------------
        print("Verifying Drive batch...")
        verified = 0

        for index in batch_indices:
            filename = f"frame_{index:06d}.pt"
            drive_file = DRIVE_CACHE / filename
            if drive_file.exists():
                verified += 1

        print(f"Verified: {verified}/{batch_total}")

        if verified != batch_total:
            raise RuntimeError("Batch verification failed. Stopping to prevent data loss.")

        # --------------------------------------------------------
        # CLEAN LOCAL CACHE
        # --------------------------------------------------------
        for local_file in LOCAL_CACHE.glob("frame_*.pt"):
            local_file.unlink()

        generated_total += verified
        elapsed = time.time() - overall_start
        total_completed = len(existing) + generated_total
        remaining = total - total_completed
        rate = generated_total / max(elapsed / 60, 1e-9)

        print(
            f"Progress: {total_completed}/{total} | "
            f"Remaining: {remaining} | "
            f"Overall generation rate: {rate:.2f} frames/min"
        )

    # ------------------------------------------------------------
    # FINAL VERIFICATION
    # ------------------------------------------------------------
    final_count = len(list(DRIVE_CACHE.glob("frame_*.pt")))
    print("\n" + "=" * 70)
    print("SEQUENCE COMPLETE")
    print("=" * 70)
    print(f"Sequence          : {SEQ}")
    print(f"Total frames      : {total}")
    print(f"Drive cache files : {final_count}")
    print(f"Generated this run: {generated_total}")

    if final_count == total:
        print("STATUS: COMPLETE")
    else:
        print(f"STATUS: INCOMPLETE ({total - final_count} missing)")

if __name__ == "__main__":
    main()
