import argparse
import logging
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import dpkt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from bytelatent.tokenizers.constants import PKT_ID, EOS_ID, OFFSET

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def run_command(command: str):
    """Executes a shell command and raises an error if it fails."""
    logging.info(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        raise

def setup_terashuf(work_dir: Path) -> Path:
    """Clones and builds terashuf if it doesn't exist."""
    terashuf_dir = work_dir / "terashuf"
    terashuf_executable = terashuf_dir / "terashuf"
    if terashuf_executable.exists():
        logging.info("terashuf executable already exists. Skipping setup.")
        return terashuf_executable

    logging.info("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_executable

def process_single_pcap(pcap_path: Path) -> bytes:
    """Processes a single PCAP file into a byte string of token IDs."""
    token_list = []
    pkt_token_array = np.array([PKT_ID], dtype=np.uint16)
    eos_token_array = np.array([EOS_ID], dtype=np.uint16)
    try:
        with open(pcap_path, 'rb') as f_in:
            pcap_reader = dpkt.pcap.Reader(f_in)
            for _, buf in pcap_reader:
                packet_tokens = np.array(list(buf), dtype=np.uint16) + OFFSET
                token_list.append(packet_tokens)
                token_list.append(pkt_token_array)
        # The EOS token marks the end of the entire PCAP file's content (our "document")
        token_list.append(eos_token_array)
        return np.concatenate(token_list).tobytes()
    except Exception as e:
        logging.warning(f"Skipping {pcap_path} due to error: {e}")
        return b''

# --- Main Pipeline Stages ---
def stage_1_read_and_convert(pcap_dir: Path, intermediate_file: Path, num_workers: int):
    """
    Stage 1: Reads all PCAP files in parallel and converts them into a single,
    shuffle-able binary file where the entire PCAP contents are separated by EOS_ID.
    """
    logging.info("--- Stage 1: Reading PCAP files and converting to intermediate format ---")
    pcap_files = sorted(list(Path(pcap_dir).rglob("*.pcap")) + list(Path(pcap_dir).rglob("*.pcapng")))
    if not pcap_files:
        raise FileNotFoundError(f"No .pcap/.pcapng files found in {pcap_dir}")

    logging.info(f"Found {len(pcap_files)} PCAP files. Processing with {num_workers} workers...")
    with Pool(processes=num_workers) as pool, open(intermediate_file, 'wb') as f_out:
        results_iterator = pool.imap_unordered(process_single_pcap, pcap_files)
        for result_bytes in tqdm(results_iterator, total=len(pcap_files), desc="Converting PCAPs"):
            if result_bytes:
                f_out.write(result_bytes)
    logging.info("--- Stage 1 Complete. ---")


def stage_2_shuffle(terashuf_exe: Path, intermediate_file: Path, shuffled_file: Path, memory_gb: int):
    """
    Stage 2: Shuffles the large intermediate file using the terashuf external tool.
    It now uses the EOS_ID as the separator, shuffling entire PCAP file contents.
    """
    logging.info("--- Stage 2: Shuffling data with terashuf (this will be slow) ---")
    # We now use the EOS_ID to separate "documents" (entire PCAP file contents).
    record_separator_bytes = EOS_ID.to_bytes(2, 'little')
    record_separator_hex = record_separator_bytes.hex()

    command = f"cat {intermediate_file} | {terashuf_exe} -s 0x{record_separator_hex} -m {memory_gb}G > {shuffled_file}"
    run_command(command)
    logging.info("--- Stage 2 Complete. ---")


def stage_3_split_and_pack(shuffled_file: Path, final_dir: Path, chunk_size: int, k_validation: int):
    """
    Stage 3: Reads the final shuffled data stream and packs it into fixed-size
    NumPy arrays for training and validation.
    """
    logging.info("--- Stage 3: Splitting and packing into final .npy files ---")
    final_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = np.fromfile(shuffled_file, dtype=np.uint16)

    num_sequences = len(all_tokens) // chunk_size
    logging.info(f"Total tokens: {len(all_tokens):,}. Total sequences: {num_sequences:,}.")

    packed_data = all_tokens[:num_sequences * chunk_size].reshape(num_sequences, chunk_size)

    # Shuffle the final sequences one last time for good measure
    np.random.shuffle(packed_data)

    validation_set = packed_data[:k_validation]
    training_set = packed_data[k_validation:]

    logging.info(f"Saving {len(training_set):,} training sequences and {len(validation_set):,} validation sequences.")
    np.save(final_dir / 'train_packed.npy', training_set)
    np.save(final_dir / 'validation_packed.npy', validation_set)
    logging.info("--- Stage 3 Complete. ---")


def main():
    default_workers = max(1, int(cpu_count() * 0.9))
    parser = argparse.ArgumentParser(
        description="A unified script to prepare PCAP data for BLT training by reading, shuffling, and splitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pcap_dir", type=str, help="Directory containing the source PCAP files.")
    parser.add_argument("output_dir", type=str, help="Directory to save all intermediate and final data.")
    parser.add_argument("--k_validation", type=int, default=12000, help="Number of sequences to use for the validation set.")
    parser.add_argument("--chunk_size", type=int, default=8192, help="The final sequence length for the model.")
    parser.add_argument("--memory_gb", type=int, default=64, help="Memory (in GB) to allocate for terashuf.")
    parser.add_argument("--num_workers", type=int, default=default_workers, help="Number of CPU cores for parallel PCAP processing.")
    parser.add_argument("--skip_stage1", action='store_true', help="Skip Stage 1 if intermediate file already exists.")
    parser.add_argument("--skip_stage2", action='store_true', help="Skip Stage 2 if shuffled file already exists.")
    parser.add_argument("--skip_stage3", action='store_true', help="Skip Stage 3 if final data already exists.")
    parser.add_argument("--cleanup", action='store_true', help="Delete large intermediate files after completion.")
    args = parser.parse_args()

    # Define file paths
    work_dir = Path.cwd()
    output_dir = Path(args.output_dir)

    # Create an output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    intermediate_file = output_dir / "all_packets.intermediate.bin"
    shuffled_file = output_dir / "all_packets.shuffled.bin"
    final_dir = output_dir / "final_packed"

    # --- Run Pipeline ---
    terashuf_executable = setup_terashuf(work_dir)

    if not args.skip_stage1:
        stage_1_read_and_convert(Path(args.pcap_dir), intermediate_file, args.num_workers)
    else:
        logging.info("Skipping Stage 1 as requested.")

    if not args.skip_stage2:
        stage_2_shuffle(terashuf_executable, intermediate_file, shuffled_file, args.memory_gb)
    else:
        logging.info("Skipping Stage 2 as requested.")

    if not args.skip_stage3:
        stage_3_split_and_pack(shuffled_file, final_dir, args.chunk_size, args.k_validation)
    else:
        logging.info("Skipping Stage 3 as requested.")

    if args.cleanup:
        logging.info("Cleaning up intermediate files...")
        intermediate_file.unlink(missing_ok=True)
        shuffled_file.unlink(missing_ok=True)

    logging.info("✅ All tasks completed successfully!")
    logging.info(f"Final data is located in: {final_dir}")

if __name__ == "__main__":
    main()
