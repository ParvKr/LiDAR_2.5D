import argparse
import logging
from pathlib import Path
import sys
from multiprocessing import Pool
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.datasets.bev_dataset import BEVDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def process_frame(args):
    dataset, idx = args
    # Accessing dataset[idx] automatically triggers the BEVProjector and saves it to disk 
    # if it's not already cached.
    _ = dataset[idx]
    return True

def main():
    parser = argparse.ArgumentParser(description="Pre-compute BEV cache across all CPU cores.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seqs", nargs="+", default=["00","01","02","03","04","05","06","07","08","09","10"])
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU cores to use.")
    args = parser.parse_args()

    for seq in args.seqs:
        seq_dir = args.dataset_root / "sequences" / seq
        if not seq_dir.exists():
            logger.warning(f"Sequence {seq} not found at {seq_dir}, skipping.")
            continue
            
        logger.info(f"Initializing Dataset for Sequence {seq}...")
        dataset = BEVDataset(seq_dir, use_cache=True)
        
        logger.info(f"Pre-computing {len(dataset)} frames using {args.workers} cores...")
        
        # We pack the dataset and index together for the multiprocessing pool
        tasks = [(dataset, i) for i in range(len(dataset))]
        
        with Pool(args.workers) as p:
            # list() forces the generator to evaluate, displaying the progress bar
            list(tqdm(p.imap_unordered(process_frame, tasks), total=len(dataset), desc=f"Seq {seq}"))

    logger.info("Cache generation complete! You can now start GPU training with 0 wait time.")

if __name__ == "__main__":
    main()
