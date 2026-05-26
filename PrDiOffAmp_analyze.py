import argparse 
import gzip 
import os 
import sys 
from collections import Counter 
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from Bio import SeqIO 
from tqdm import tqdm


#   Opening a FASTQ (plain or gzipped) and yield SeqRecord objects.
def _open_maybe_gz(path: Path) -> Iterable[SeqIO.SeqRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"FASTQ file not found: {path}")

    # Choose the appropriate opener
    if path.suffix.lower() in {".gz", ".gzip"}:
        # ``gzip.open`` returns a text‑mode file object when mode="rt"
        handle = gzip.open(path, "rt")
        return SeqIO.parse(handle, "fastq")
    else:
        return SeqIO.parse(str(path), "fastq")
    

# Reading fastq file and merging pairs
def read_fastq(r1_path: Path, r2_path: Path) -> List[Tuple[str, int]]: 
    r1_iter = _open_maybe_gz(r1_path)
    r2_iter = _open_maybe_gz(r2_path)

    merged = []
    for rec1, rec2 in tqdm(zip(r1_iter, r2_iter),
                        total=None,
                        desc="Reading FASTQ pairs",
                        unit="pair"):
        if rec1.id != rec2.id:
            # In many pipelines the IDs are identical; if not we still keep
            # the first part before any whitespace.
            # also splitting the part where the number in the pair is indicated (1/2)
            id1 = rec1.id.split()[0].split('/')[0]
            id2 = rec2.id.split()[0].split('/')[0]
            if id1 != id2:
                raise ValueError(
                    f"Read IDs do not match: {rec1.id} vs {rec2.id}"
                )
            read_id = id1
        else:
            read_id = rec1.id

        merged_len = len(rec1.seq) + len(rec2.seq)
        merged_seq = (rec1.seq) + (rec2.seq)
        merged.append((read_id, merged_len, merged_seq))

    # sanity check – make sure we consumed both files completely
    # (the zip iterator stops at the shortest, so we need to verify lengths)
    r1_count = sum(1 for _ in _open_maybe_gz(r1_path))
    r2_count = sum(1 for _ in _open_maybe_gz(r2_path))
    if r1_count != r2_count:
        raise ValueError(
            f"FASTQ files have different number of reads: {r1_count} vs {r2_count}"
        )
    return merged


# Reading primers
def read_primers(fasta_path: Path) -> Tuple[str, str]: 
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if len(records) != 2:
        raise ValueError(
            f"Primer FASTA must contain exactly two records, found {len(records)}"
        )
    # Determine which is forward / reverse by the header name
    forward = None
    reverse = None
    for rec in records:
        name = rec.id.lower()
        if "forward" in name:
            forward = str(rec.seq).upper()
        elif "reverse" in name:
            reverse = str(rec.seq).upper()
        else:
            # fallback – assume the first is forward, second reverse
            if forward is None:
                forward = str(rec.seq).upper()
            else:
                reverse = str(rec.seq).upper()
    if forward is None or reverse is None:
        raise ValueError("Could not identify forward and reverse primers.")
    return forward, reverse


# Distance for detecting number of mismatches
def _hamming_distance(s1: str, s2: str) -> int:
    """Return the Hamming distance between two equal‑length strings."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length for Hamming distance")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


# Checking primer matches
def _primer_match_at_end(seq: str, primer: str, max_mismatches: int = 2) -> bool:
    tail = seq[-len(primer) :].upper()
    if _hamming_distance(tail, primer) <= max_mismatches:
        return True
    rc = str(Seq(primer).reverse_complement())
    return _hamming_distance(tail, rc) <= max_mismatches


# Detecting primer dimers
def detect_primer_dimers(
    merged_reads: List[Tuple[str, int]],
    forward_primer: str,
    reverse_primer: str,
    max_dimer_len: int = 100,
    max_mismatches: int = 2,
) -> List[str]:

    dimers = []
    for read_id, length, merged_seq in merged_reads:
        # Fast heuristic: any read shorter than the cutoff is considered a dimer.
        #if length <= max_dimer_len*2:
        #    dimers.append(read_id)
        # checking for the length and mismatches with primers
        if length <= max_dimer_len*2:
            if (_primer_match_at_end(merged_seq, forward_primer, max_mismatches) and _primer_match_at_end(merged_seq[::-1], reverse_primer, max_mismatches)):
                dimers.append(read_id)
    return dimers

# Detecting sequencing shorter or longer than expected length (600=2*read_len)
def detect_off_targets( merged_reads: List[Tuple[str, int]], expected_len: int = 600, tolerance: int = 50, ) -> Tuple[List[str], List[str]]:
    lower = expected_len - tolerance
    upper = expected_len + tolerance
    short_reads, long_reads = [], []
    for read_id, length, _ in merged_reads:
        if length < lower:
            short_reads.append(read_id)
        elif length > upper:
            long_reads.append(read_id)
    return short_reads, long_reads


# Taking all the statistics (counts) together
def summarise_counts( sample_id: str, total_reads: int, primer_dimer_ids: List[str], short_off_ids: List[str], long_off_ids: List[str], ) -> pd.DataFrame: 
    primer_dimer_count = len(primer_dimer_ids)
    short_off_count = len(short_off_ids)
    long_off_count = len(long_off_ids)

    valid_amplicon_count = (
        total_reads - primer_dimer_count - short_off_count - long_off_count
    )
    primer_dimer_pct = 100.0 * primer_dimer_count / total_reads if total_reads else 0.0

    df = pd.DataFrame(
        {
            "sample_id": [sample_id],
            "total_reads": [total_reads],
            "primer_dimer_count": [primer_dimer_count],
            "primer_dimer_percentage": [primer_dimer_pct],
            "short_offtarget_count": [short_off_count],
            "long_offtarget_count": [long_off_count],
            "valid_amplicon_count": [valid_amplicon_count],
        }
    )
    return df


# Plotting the statistics
def plot_length_distribution( merged_reads: List[Tuple[str, int]], primer_dimer_ids: List[str], short_off_ids: List[str], long_off_ids: List[str], expected_len: int = 600, tolerance: int = 50, output_path: Path | None = None, ) -> plt.Figure: 
    lengths = np.array([ln for _, ln, _ in merged_reads])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins=range(0, max(lengths) + 20, 20), color="#6baed6", edgecolor="black")
    ax.set_xlabel("Merged read length (bp)")
    ax.set_ylabel("Read count")
    ax.set_title("Read‑length distribution with primer‑dimer / off‑target zones")

    # Shade regions
    max_dimer_len = 100*2
    ax.axvspan(0, max_dimer_len, color="#ff7f0e", alpha=0.2, label="Primer‑dimer zone")
    lower = expected_len - tolerance
    upper = expected_len + tolerance
    ax.axvspan(lower, upper, color="#2ca02c", alpha=0.2, label="Expected amplicon zone")
    ax.axvspan(upper, max(lengths) + 10, color="#d62728", alpha=0.2, label="Long off‑target zone")
    ax.axvspan(0, lower, color="#9467bd", alpha=0.2, label="Short off‑target zone")

    ax.legend()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
    return fig


# General workflow
def run_workflow( sample_id: str, r1_path: Path, r2_path: Path, primer_fasta: Path, expected_len: int = 600, length_tolerance: int = 50, max_dimer_len: int = 100, quality_threshold: int = 30, max_mismatches: int = 2, plot_path: Path | None = None, ) -> pd.DataFrame: 

    # ------------------------------------------------------------------
    # 1. Load reads
    # ------------------------------------------------------------------
    merged_reads = read_fastq(r1_path, r2_path)
    total_reads = len(merged_reads)

    # ------------------------------------------------------------------
    # 2. Load primers
    # ------------------------------------------------------------------
    forward_primer, reverse_primer = read_primers(primer_fasta)

    # ------------------------------------------------------------------
    # 3. Primer‑dimer detection
    # ------------------------------------------------------------------
    primer_dimer_ids = detect_primer_dimers(
        merged_reads,
        forward_primer,
        reverse_primer,
        max_dimer_len=max_dimer_len,
        max_mismatches=max_mismatches,
    )

    # ------------------------------------------------------------------
    # 4. Off‑target detection
    # ------------------------------------------------------------------
    short_off_ids, long_off_ids = detect_off_targets(
        merged_reads,
        expected_len=expected_len,
        tolerance=length_tolerance,
    )

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    summary_df = summarise_counts(
        sample_id,
        total_reads,
        primer_dimer_ids,
        short_off_ids,
        long_off_ids,
    )

    # ------------------------------------------------------------------
    # 6. Plot (optional)
    # ------------------------------------------------------------------
    if plot_path:
        plot_length_distribution(
            merged_reads,
            primer_dimer_ids,
            short_off_ids,
            long_off_ids,
            expected_len=expected_len,
            tolerance=length_tolerance,
            output_path=plot_path,
        )
    return summary_df


# Defining parser for reading command line argument
def arg_parser():
    if len(sys.argv) == 1:
        sys.exit(
            "No arguments provided. You need to provide path to the files (see -h)"
        )
    parser = argparse.ArgumentParser(
        description="Mishpokhe: self-supervised discovery of functional clusters"
    )
    parser.add_argument( "-i", "--sample-id", required=True, help="Sample identifier (used in the summary table)", ) 
    parser.add_argument( "-1", "--r1", required=True, type=Path, help="Path to forward FASTQ (R1). Supports .gz", ) 
    parser.add_argument( "-2", "--r2", required=True, type=Path, help="Path to reverse FASTQ (R2). Supports .gz", ) 
    parser.add_argument( "-p", "--primers", required=True, type=Path, help="FASTA file containing forward and reverse primers", ) 
    parser.add_argument( "-o", "--out", type=Path, default=Path("summary.tsv"), help="Path to write the tab‑separated summary (default: summary.tsv)", ) 
    parser.add_argument( "--expected-len", type=int, default=600, help="Expected length range (default: 600)", ) 
    parser.add_argument( "--tolerance", type=int, default=50, help="Length tolerance around the expected length (default: 50)", ) 
    parser.add_argument( "--max-dimer-len", type=int, default=100, help="Maximum length to consider a read a primer‑dimer (default: 100)", ) 
    parser.add_argument( "--max-mismatches", type=int, default=2, help="Maximum mismatches allowed when checking primer ends (default: 2)", ) 
    parser.add_argument( "--plot", type=Path, default=None, help="If supplied, write a histogram PNG/PDF to this path", )

    global args
    args, unknown = parser.parse_known_args()
    print(vars(args))
    # checking if filepaths exist
    for argument in vars(args):
        arg_path = getattr(args, argument)
        if not os.path.exists((arg_path)):
            # looking at arguments corresponding to files that must exist
            if argument in [
                "r1",
                "r2",
                "primers"
            ]:
                sys.exit(f"{arg_path} not found")
            # now checking if there is such argument in non-path arguments
            if argument not in [
                "sample_id",
                "out",
                "expected_len",
                "tolerance",
                "max_dimer_len",
                "max_mismatches",
                "plot"
            ]:
                sys.exit(f"{argument} not found")


def main() -> None: 
    arg_parser()
    try:
        summary = run_workflow(
            sample_id=args.sample_id,
            r1_path=args.r1,
            r2_path=args.r2,
            primer_fasta=args.primers,
            expected_len=args.expected_len,
            length_tolerance=args.tolerance,
            max_dimer_len=args.max_dimer_len,
            max_mismatches=args.max_mismatches,
            plot_path=args.plot,
        )
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    # Writing the summary table
    summary.to_csv(args.out, sep="\t", index=False)
    print(f"Summary written to {args.out}")


if __name__ == "__main__": main()