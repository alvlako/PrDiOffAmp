import random
from Bio.Seq import Seq

# -------------------------------------------------
# USER‑CONFIGURABLE PARAMETERS
# -------------------------------------------------
N_AMPLICONS = 100          # regular amplicons
N_SHORT_DIMERS = 10        # primer‑dimers > 100 bp
N_LONG_DIMERS  = 10        # primer‑dimers > 450 bp = off-target amplification

MEAN_LEN   = 400           # mean amplicon length (incl. primers)
STD_LEN    = 15            # ~±50 bp (95 % CI ≈ 2 × STD)

FWD_PRIMER = "ACGTACGTACGT"
REV_PRIMER = "TGCATGCATGCA"

READ_LEN   = 300           # length of each Illumina read (R1 & R2)

OUT_R1 = "R1.fastq"        # forward‑read FASTQ
OUT_R2 = "R2.fastq"        # reverse‑read FASTQ

# Quality‑score generation parameters
QUAL_MEAN = 30             # average Phred quality (Q30 ≈ 99.9 % accuracy)
QUAL_STD  = 5              # spread of the quality scores
QUAL_MIN  = 2              # lowest allowed Q‑score
QUAL_MAX  = 40             # highest allowed Q‑score
# -------------------------------------------------


def random_seq(length: int) -> str:
    """Return a random DNA string of the requested length."""
    return ''.join(random.choices('ACGT', k=length))


def make_amplicon(target_len: int) -> str:
    """
    Build a normal amplicon:
        <FWD_PRIMER><random interior><REV_PRIMER>
    """
    interior_len = target_len - len(FWD_PRIMER) - len(REV_PRIMER)
    interior = random_seq(interior_len)
    return FWD_PRIMER + interior + REV_PRIMER


def make_primer_dimer(min_len: int, max_len: int) -> str:
    """
    Build a primer‑dimer‑like sequence.
    The simplest model is:
        <FWD_PRIMER><random filler><REV_PRIMER><random filler>
    The total length is forced into the interval [min_len, max_len].
    """
    # Reserve space for the two primers
    filler_len = max(min_len, max_len) - len(FWD_PRIMER) - len(REV_PRIMER)
    # Split filler into two parts (before & after the reverse primer)
    part1_len = filler_len // 2
    part2_len = filler_len - part1_len

    part1 = random_seq(part1_len)
    part2 = random_seq(part2_len)

    seq = FWD_PRIMER + part1 + REV_PRIMER + part2

    # If the generated sequence is still shorter than min_len, pad at the end
    while len(seq) < min_len:
        seq += random.choice('ACGT')
    # If it is longer than max_len, truncate the tail
    if len(seq) > max_len:
        seq = seq[:max_len]

    return seq


def phred_quality_string(length: int,
                         mean: int = QUAL_MEAN,
                         std:  int = QUAL_STD,
                         qmin: int = QUAL_MIN,
                         qmax: int = QUAL_MAX) -> str:
    """
    Generate a random Phred quality string of *length* characters.
    Scores are drawn from a normal distribution (mean/std) and clipped
    to the interval [qmin, qmax] before conversion to ASCII (Phred+33).
    """
    quals = []
    for _ in range(length):
        q = int(random.gauss(mean, std))
        q = max(qmin, min(qmax, q))          # clip
        quals.append(chr(q + 33))            # Phred+33 → ASCII
    return ''.join(quals)


def write_fastq_pair(r1_h, r2_h,
                     seq_id: str,
                     fwd_seq: str,
                     rev_seq: str, 
                     actual_len: int):
    """
    Write one paired‑end record to the two FASTQ handles.
    The quality strings are generated on‑the‑fly.
    """
    # quality strings for normal reads
    qual_r1 = phred_quality_string(actual_len)
    qual_r2 = phred_quality_string(actual_len)

    r1_h.write(f"@{seq_id}/1\n{fwd_seq}\n+\n{qual_r1}\n")
    r2_h.write(f"@{seq_id}/2\n{rev_seq}\n+\n{qual_r2}\n")


def extract_reads_from_amplicon(amplicon: str, seq_id: str, actual_len: int,
                               r1_h, r2_h):
    """
    Given a full amplicon (or dimer) sequence, pull out the forward
    (first READ_LEN bases) and reverse‑complement of the last READ_LEN bases,
    then write them as a paired‑end record.
    """
    # Forward read – first READ_LEN bases (as‑is)
    r1_seq = amplicon[:actual_len]
    # Reverse read – reverse‑complement of the last READ_LEN bases
    r2_raw = amplicon[-actual_len:]
    r2_seq = str(Seq(r2_raw).reverse_complement())

    write_fastq_pair(r1_h, r2_h, seq_id, r1_seq, r2_seq, actual_len)


def generate():
    with open(OUT_R1, "w") as fh_r1, open(OUT_R2, "w") as fh_r2:
        # -------------------------------------------------
        # 1️⃣  Regular amplicons
        # -------------------------------------------------
        for i in range(1, N_AMPLICONS + 1):
            # Choose a realistic length and clamp to 350‑450 bp
            target_len = int(random.gauss(MEAN_LEN, STD_LEN))
            target_len = max(350, min(450, target_len))

            amplicon = make_amplicon(target_len)
            seq_id = f"amplicon_{i}"
            actual_len = READ_LEN
            extract_reads_from_amplicon(amplicon, seq_id, actual_len, fh_r1, fh_r2)

        # -------------------------------------------------
        # 2️⃣  Short primer‑dimers (<100 bp)
        # -------------------------------------------------
        for i in range(1, N_SHORT_DIMERS + 1):
            # We ask for a length somewhere between 10 and 100 bp
            dimer = make_primer_dimer(min_len=10, max_len=100)
            seq_id = f"short_dimer_{i}"
            actual_len = len(dimer)
            extract_reads_from_amplicon(dimer, seq_id, actual_len, fh_r1, fh_r2)

        # -------------------------------------------------
        # 3️⃣  Long primer‑dimers (>450 bp)
        # -------------------------------------------------
        for i in range(1, N_LONG_DIMERS + 1):
            # Length between 451 and 550 bp (feel free to enlarge)
            dimer = make_primer_dimer(min_len=451, max_len=550)
            seq_id = f"long_dimer_{i}"
            actual_len = len(dimer)
            extract_reads_from_amplicon(dimer, seq_id, actual_len, fh_r1, fh_r2)

    print(f" Finished –")
    print(f"   • {N_AMPLICONS} regular amplicon pairs")
    print(f"   • {N_SHORT_DIMERS} short primer‑dimer pairs (<100 bp)")
    print(f"   • {N_LONG_DIMERS}  long primer‑dimer pairs (>450 bp)")
    print(f"   written to '{OUT_R1}' and '{OUT_R2}'.")

generate()