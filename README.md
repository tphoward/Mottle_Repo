# Mottle

Mottle is a bioinformatics tool for calculating an accurate substitution distance between highly divergent nucleic acid sequences.

## Installation
    git clone https://github.com/tphoward/Mottle_Repo.git
    cd Mottle_Repo
    conda env create -f environment.yml
    conda activate mottle

## Usage
    usage: mottle.py [-h] [-i IN1] [-I IN2] [-o OUT] [--chunk_size CHUNK_SIZE] [--nchunks NCHUNKS] [--guide_chunks GUIDE_CHUNKS]
                     [--window_shape {boxcar}] [--encodings [{MAFFT,SQUARE,AG,PUPY,TRANS,NUC.4.4,BLOSUM62,PROPS} ...]] [--detr {constant}]
                     [--norm_level NORM_LEVEL] [--index {Flat}] [--reduct {mip}] [--nmatch [NMATCH ...]] [--sample {all}] [--filt_size FILT_SIZE]
                     [--max_pass MAX_PASS] [--cut_thres CUT_THRES] [--samp_width SAMP_WIDTH] [--min_samps MIN_SAMPS] [--ntrees NTREES]
                     [--nleaves NLEAVES] [--learn_rate LEARN_RATE] [--subsamp SUBSAMP] [--binpow BINPOW] [--learn_mult LEARN_MULT]
                     [--reltol RELTOL] [--maxiter MAXITER] [--binthres BINTHRES] [--prior_size PRIOR_SIZE] [--ncpu NCPU] [-v]
                     [in1] [in2] [out]

    Mottle - Pairwise substitution distance at high divergences

    positional arguments:
      in1                   Input fasta file 1 (default: None)
      in2                   Input fasta file 2 (default: -)
      out                   Output file for final value (default: -)

    optional arguments:
      -h, --help            show this help message and exit
      -i IN1, --in1 IN1, --in IN1
                            Input fasta file 1 (default: None)
      -I IN2, --in2 IN2, -j IN2
                            Input fasta file 2 (default: None)
      -o OUT, --out OUT     Output file for final value (default: None)
      --chunk_size CHUNK_SIZE
                            Size of each chunk in base pairs (default: 18)
      --nchunks NCHUNKS     Total number of chunks used for search window (default: 20)
      --guide_chunks GUIDE_CHUNKS
                            Number of chunks directly aligned, without InDels taken into account (default: 5)
      --window_shape {boxcar}
                            Shape of search window (default: boxcar)
      --encodings [{MAFFT,SQUARE,AG,PUPY,TRANS,NUC.4.4,BLOSUM62,PROPS} ...]
                            Encodings used for sequence search (default: ['MAFFT'])
      --detr {constant}     GC content correction mode (default: constant)
      --norm_level NORM_LEVEL
                            Exponent used for normalisation (default: 1)
      --index {Flat}        Faiss search index type (default: Flat)
      --reduct {mip}        Faiss search reduction type (default: mip)
      --nmatch [NMATCH ...]
                            Number of nearest neighbours returned per site (default: [1])
      --sample {all}        Subsample fraction (default: all)
      --filt_size FILT_SIZE
                            Size of window for filtering near-origin alignment gaps (default: 9)
      --max_pass MAX_PASS   Maximum windows kept after filtering (default: 3000)
      --cut_thres CUT_THRES
                            Maximum sub-window divergence before alignment is cut (default: 2)
      --samp_width SAMP_WIDTH
                            Sub-window sample size (default: 100)
      --min_samps MIN_SAMPS
                            Minimum number of sub-windows samples so that alignment is not discarded (default: 150)
      --ntrees NTREES       LightGBM number of trees for true identity estimation (default: 100)
      --nleaves NLEAVES     LightGBM number of leaves (default: 31)
      --learn_rate LEARN_RATE
                            LightGBM learn rate (default: 0.1)
      --subsamp SUBSAMP     LightGBM subsample proportion (default: 1)
      --binpow BINPOW       Exponent for cluster discretisation (default: 64)
      --learn_mult LEARN_MULT
                            Tensorflow learning multiplier (default: 0.001)
      --reltol RELTOL       Tensorflow relative tolerace stopping codition (default: 1e-20)
      --maxiter MAXITER     Tensorflow maximum number of gradient descent iterations (default: 100)
      --binthres BINTHRES   Threshold for sequence inclusion into homology cluster (default: 0.75)
      --prior_size PRIOR_SIZE
                            Bias value for distance calculations where few alignments pass the filtering stage (default: 10)
      --ncpu NCPU           Number of cpu threads for intensive tasks (default: 4)
      -v, --verbose         Verbosity toggle (default: False)

    Mottle Copyright 2023 Newcastle University. All Rights Reserved. Authors: Alisa Prusokiene, Neil Boonham, Adrian Fox, and Thomas P. Howard.
    The initial repository for this software is located at https://github.com/tphoward/Mottle_Repo.
