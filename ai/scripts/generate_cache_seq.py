import os
import shutil
import time
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# PROJECT
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ai"))

from data.datasets.bev_dataset import BEVDataset

def main():
    parser = argparse.ArgumentParser(description="Generate cache and backup to a secondary drive.")
    parser.add_argument("--seqs", nargs="+", default=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"], help="List of sequences to process")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/drive/MyDrive/semantickitti/dataset"))
    parser.add_argument("--local-cache-root", type=Path, default=Path("/content/bev_cache"))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    for SEQ in args.seqs:
        DATASET_ROOT = args.dataset_root
        SEQUENCE_DIR = DATASET_ROOT / "sequences" / SEQ

        # ------------------------------------------------------------
        # CACHE LOCATION
        # ------------------------------------------------------------
        # Force the BEVDataset to write its cache here initially
        os.environ["BEV_CACHE_DIR"] = str(args.local_cache_root.parent)

        # ------------------------------------------------------------
        # DATASET INSTANTIATION
        # ------------------------------------------------------------
        try:
            dataset = BEVDataset(SEQUENCE_DIR, use_cache=True)
        except Exception as e:
            print(f"Error loading dataset for sequence {SEQ}: {e}")
            continue

        # ------------------------------------------------------------
        # DIRECTORIES
        # ------------------------------------------------------------
        proj = dataset.projector
        CONFIG_NAME = f"res_{proj.resolution}_x_{proj.x_min}_{proj.x_max}_y_{proj.y_min}_{proj.y_max}_z_{proj.z_min}_{proj.z_max}"

        DRIVE_CACHE = (
            SEQUENCE_DIR
            / ".bev_cache"
            / CONFIG_NAME
        )

        # Dynamically grab the exact hashed path the dataset will use
        LOCAL_CACHE = dataset.cache_dir / CONFIG_NAME

        DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
        LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

        BATCH_SIZE = args.batch_size
        total = len(dataset)

        # ------------------------------------------------------------
        # FIND EXISTING FILES TO SKIP
        # ------------------------------------------------------------
        existing_drive = {p.name for p in DRIVE_CACHE.glob("frame_*.pt")}
        existing_local = {p.name for p in LOCAL_CACHE.glob("frame_*.pt")}

        print("=" * 70)
        print(f"SEQUENCE {SEQ} CACHE GENERATION")
        print("=" * 70)
        print(f"Total frames     : {total}")
        print(f"Already on Drive : {len(existing_drive)}")
        print(f"Already in Local : {len(existing_local)}")
        print(f"Drive cache      : {DRIVE_CACHE}")
        print(f"Local cache      : {LOCAL_CACHE}")
        print(f"Batch size       : {BATCH_SIZE}")
        print()

        # ------------------------------------------------------------
        # GENERATE IN BATCHES
        # ------------------------------------------------------------
        # Only process frames that are NOT already safely on Drive
        remaining_indices = [i for i in range(total) if f"frame_{i:06d}.pt" not in existing_drive]
        overall_start = time.time()
        generated_total = 0

        for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
            batch_indices = remaining_indices[batch_start:batch_start + BATCH_SIZE]
            batch_number = (batch_start // BATCH_SIZE) + 1
            batch_total = len(batch_indices)

            print(f"\nBatch {batch_number} | {batch_total} frames")

            batch_start_time = time.time()

            for index in tqdm(batch_indices, desc="Generating Cache"):
                filename = f"frame_{index:06d}.pt"
                local_file = LOCAL_CACHE / filename
                
                # If it already exists locally from a crash, don't waste time regenerating!
                if not local_file.exists():
                    _ = dataset[index]

            generation_time = (time.time() - batch_start_time)
            local_files = list(LOCAL_CACHE.glob("frame_*.pt"))

            print(f"Local cache files ready: {len(local_files)}")
            print(f"Generation time: {generation_time:.1f}s")

            print("Copying batch to Google Drive...")
            copy_start = time.time()
            copied = 0

            for local_file in tqdm(local_files, desc="Copying to Drive"):
                drive_file = DRIVE_CACHE / local_file.name
                shutil.copy2(local_file, drive_file)
                copied += 1

            copy_time = time.time() - copy_start
            print(f"Copied {copied} files in {copy_time:.1f}s")

            print("Verifying Drive batch...")
            verified = 0

            for index in tqdm(batch_indices, desc="Verifying Files"):
                filename = f"frame_{index:06d}.pt"
                drive_file = DRIVE_CACHE / filename
                if drive_file.exists():
                    verified += 1

            print(f"Verified: {verified}/{batch_total}")

            if verified != batch_total:
                raise RuntimeError("Batch verification failed. Stopping to prevent data loss.")

            for local_file in LOCAL_CACHE.glob("frame_*.pt"):
                local_file.unlink()

            generated_total += verified
            elapsed = time.time() - overall_start
            
            # The actual completed count includes the ones we skipped
            total_completed = len(existing_drive) + generated_total
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
        print(f"SEQUENCE {SEQ} COMPLETE")
        print("=" * 70)
        print(f"Sequence          : {SEQ}")
        print(f"Total frames      : {total}")
        print(f"Drive cache files : {final_count}")
        print(f"Generated this run: {generated_total}")

        if final_count == total:
            print("STATUS: COMPLETE\n")
        else:
            print(f"STATUS: INCOMPLETE ({total - final_count} missing)\n")

if __name__ == "__main__":
    main()
