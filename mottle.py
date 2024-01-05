#!/usr/bin/env python
# coding: utf-8

# In[3]:


import argparse
import numpy as np


# In[4]:


parser = argparse.ArgumentParser(
    prog = 'python mottle.py',
    description = 'Mottle - Pairwise substitution distance at high divergences',
    epilog = 'Mottle Copyright 2023 Newcastle University. All Rights Reserved.\n' +
    'Authors: Alisa Prusokiene, Neil Boonham, Adrian Fox, and Thomas P. Howard.\n' +
    'The initial repository for this software is located at https://github.com/tphoward/Mottle_Repo.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('in1_p', nargs='?', type=argparse.FileType('r'), default=None, metavar='in1', help='Input fasta file 1')
parser.add_argument('in2_p', nargs='?', type=argparse.FileType('r'), default='-', metavar='in2', help='Input fasta file 2')
parser.add_argument('out_p', nargs='?', type=argparse.FileType('w'), default='-', metavar='out', help='Output file for final value')
parser.add_argument('-i', '--in1', '--in', type=argparse.FileType('r'), default=None, help='Input fasta file 1')
parser.add_argument('-I', '--in2', '-j', type=argparse.FileType('r'), default=None, help='Input fasta file 2')
parser.add_argument('-o', '--out', type=argparse.FileType('w'), default=None, help='Output file for final value')
parser.add_argument('--chunk_size', type=np.uint, default=18, help='Size of each chunk in base pairs')
parser.add_argument('--nchunks', type=np.uint, default=20, help='Total number of chunks used for search window')
parser.add_argument('--guide_chunks', type=np.uint, default=5, help='Number of chunks directly aligned, without InDels taken into account')
parser.add_argument('--window_shape', type=str, default='boxcar', choices=['boxcar'], help='Shape of search window')
parser.add_argument('--encodings', type=str, nargs='*', default=['MAFFT'],
                    choices=['MAFFT', 'SQUARE', 'AG', 'PUPY', 'TRANS', 'NUC.4.4', 'BLOSUM62', 'PROPS'],
                    help='Encodings used for sequence search')
parser.add_argument('--detr', type=str, default='constant', choices=['constant'], help='GC content correction mode')
parser.add_argument('--norm_level', type=np.uint, default=1, help='Exponent used for normalisation')
parser.add_argument('--index', type=str, default='Flat', choices=['Flat'], help='Faiss search index type')
parser.add_argument('--reduct', type=str, default='mip', choices=['mip'], help='Faiss search reduction type')
parser.add_argument('--nmatch', type=np.uint, nargs='*', default=[1], help='Number of nearest neighbours returned per site')
parser.add_argument('--sample', type=str, default='all', choices=['all'], help='Subsample fraction')
parser.add_argument('--filt_size', type=np.uint, default=9, help='Size of window for filtering near-origin alignment gaps')
parser.add_argument('--max_pass', type=np.uint, default=3000, help='Maximum windows kept after filtering')
parser.add_argument('--cut_thres', type=np.uint, default=2, help='Maximum sub-window divergence before alignment is cut')
parser.add_argument('--samp_width', type=np.uint, default=100, help='Sub-window sample size')
parser.add_argument('--min_samps', type=np.uint, default=150, help='Minimum number of sub-windows samples so that alignment is not discarded')
parser.add_argument('--ntrees', type=np.uint, default=100, help='LightGBM number of trees for true identity estimation')
parser.add_argument('--nleaves', type=np.uint, default=31, help='LightGBM number of leaves')
parser.add_argument('--learn_rate', type=np.uint, default=0.1, help='LightGBM learn rate')
parser.add_argument('--subsamp', type=np.uint, default=1, help='LightGBM subsample proportion')
parser.add_argument('--binpow', type=np.uint, default=64, help='Exponent for cluster discretisation')
parser.add_argument('--learn_mult', type=float, default=0.001, help='Tensorflow learning multiplier')
parser.add_argument('--reltol', type=float, default=1e-20, help='Tensorflow relative tolerace stopping codition')
parser.add_argument('--maxiter', type=np.uint, default=100, help='Tensorflow maximum number of gradient descent iterations')
parser.add_argument('--binthres', type=float, default=0.75, help='Threshold for sequence inclusion into homology cluster')
parser.add_argument('--prior_size', type=np.uint, default=10, help='Bias value for distance calculations where few alignments pass the filtering stage')
parser.add_argument('--ncpu', type=np.uint, default=4, help='Number of cpu threads for intensive tasks')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbosity toggle')


# In[ ]:


args = parser.parse_args()
infile1 = args.in1 if args.in1 else args.in1_p
infile2 = args.in2 if args.in2 else args.in2_p
outfile = args.out if args.out else args.out_p
if infile1 is None:
    parser.print_help()
    exit()

chunk_size = args.chunk_size
nchunks = args.nchunks
guide_chunks = args.guide_chunks
window_size = nchunks * chunk_size
guide_size = guide_chunks * chunk_size
window_shape = args.window_shape
encodings = args.encodings
detr = args.detr
norm_level = args.norm_level
index = args.index
reduct = args.reduct
nmatch = args.nmatch
sample = args.sample
filt_size = args.filt_size
max_pass = args.max_pass
cut_thres = args.cut_thres
samp_width = args.samp_width
min_samps = args.min_samps
ntrees = args.ntrees
nleaves = args.nleaves
learn_rate = args.learn_rate
subsamp = args.subsamp
binpow = args.binpow
learn_mult = args.learn_mult
reltol = args.reltol
maxiter = args.maxiter
binthres = args.binthres
prior_size = args.prior_size
ncpu = args.ncpu
verbose = args.verbose

eps = np.finfo(np.float32).resolution


# In[239]:


import sys, re

from collections import defaultdict
from io import StringIO
from Bio import SeqIO
from scipy import signal, fft, stats
from numpy import ma
from numpy.lib.stride_tricks import as_strided
import faiss
import parasail
from tqdm import tqdm
import lightgbm as lgb
import tensorflow as tf
import tensorflow_probability as tfp


# In[ ]:


def strtoseq(string):
    """Convert string to numpy array."""
    return np.fromiter(string, "U1")

def guessab(seq):
    """Guess sequence alphabet."""
    dna = ('A', 'T', 'C', 'G', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'X', 'N')
    rna = ('A', 'U', 'C', 'G', 'Y', 'R', 'W', 'S', 'K', 'M', 'D', 'V', 'H', 'B', 'X', 'N')
    prot = np.array([
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
            'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X', '*'], dtype='<U1')
    code = np.unique(seq)
    alphabet = 'NONE'
    if np.isin(code, dna).all():
        alphabet = 'DNA'
    elif np.isin(code, rna).all():
        alphabet = 'RNA'
    elif np.isin(code, prot).all():
        alphabet = 'PROT'
    return alphabet

def read_fasta(path, alphabet='guess'):
    """Load sequences from fasta. Return data structs."""
    dtype = np.dtype([
        ('id', (np.str_, 20)), ('name', (np.str_, 50)), ('desc', (np.str_, 120)),
        ('alphabet', (np.str_, 8)), ('start', int), ('end', int)])
    meta = []
    structs = []
    start = 0
    for data in SeqIO.parse(path, 'fasta'):
        end = start + len(data.seq)
        name = re.search('\A[^(,)]*', data.description.split(' ', 1)[-1]).group(0)
        seq = strtoseq(str(data.seq))
        alphabet = guessab(seq) if alphabet=='guess' else alphabet
        meta.append(np.array((data.id, name, data.description, alphabet, start, end), dtype=dtype))
        structs.extend(seqtorecs(seq, start))
        start = end
    return np.asarray(meta), structs

def get_codex(encoding):
    """Return codex dictionary for selected encoding."""
    defval = 0+0j
    sqhf = np.sqrt(0.5)
    sqhfj = sqhf*1j
    if encoding == 'NA':
        alphabet = np.array([
            'A', 'T', 'U', 'C', 'G', 'Y', 'R', 'W',
            'S', 'K', 'M', 'D', 'V', 'H', 'B', 'X', 'N'], dtype='<U1')
        encoded = np.array([
            'A', 'T', 'T', 'C', 'G', 'Y', 'R', 'W',
            'S', 'K', 'M', 'D', 'V', 'H', 'B', 'N', 'N'], dtype='<U1')
        defval = 'N'
    elif encoding == 'COMPL':
        alphabet = np.array([
            'A', 'T', 'U', 'C', 'G', 'Y', 'R', 'W',
            'S', 'K', 'M', 'D', 'V', 'H', 'B', 'X', 'N'], dtype='<U1')
        encoded = np.array([
            'T', 'A', 'A', 'G', 'C', 'R', 'Y', 'W',
            'S', 'M', 'K', 'H', 'B', 'D', 'V', 'N', 'N'], dtype='<U1')
        defval = 'N'
    elif encoding == 'SQUARE':
        alphabet = np.array([
            'A', 'T', 'U', 'C', 'G', 'Y', 'R', 'W',
            'S', 'K', 'M', 'D', 'V', 'H', 'B', 'X', 'N'], dtype='<U1')
        encoded = np.array([
            1-1j, 1+1j, 1+1j, -1+1j, -1-1j], dtype=np.complex64)
    elif encoding == 'MAFFT':
        alphabet = np.array([
            'A', 'T', 'U', 'C', 'G',
            'Y', 'R', 'W', 'S',
            'K', 'M', 'D', 'V',
            'H', 'B', 'X', 'N'], dtype='<U1')
        encoded = np.array([
            1j, -1j, -1j, -1, 1,
            -sqhf-sqhfj, sqhf+sqhfj, 0+0j, 0+0j,
            sqhf-sqhfj, -sqhf+sqhfj, 1+0j, 0+1j,
            -1+0j, 0-1j, 0+0j, 0+0j], dtype=np.complex64)
    elif encoding=='AG':
        alphabet = np.array(['A', 'T', 'U', 'C', 'G'], dtype='<U1')
        encoded = np.array((1, 0, 0, 0, 1j), dtype=np.complex64)
    elif encoding=='PUPY':
        alphabet = np.array(['A', 'T', 'U', 'C', 'G'], dtype='<U1')
        encoded = np.array((1, 1j, 1j, 1j, 1), dtype=np.complex64)
    elif encoding == 'TRANS':
        alphabet = np.array(['A', 'T', 'C', 'G'], dtype='<U1')
        encoded = np.array([
            0.8944271 +0.4472137j , -0.4472138 -0.8944271j ,
           -0.8944272 -0.44721368j,  0.44721365+0.8944272j ], dtype=np.complex64)
    elif encoding == 'NUC.4.4':
        alphabet = np.array([
            'A', 'T', 'G', 'C', 'S', 'W', 'R', 'Y',
            'K', 'M', 'B', 'V', 'H', 'D', 'N'], dtype='<U1')
        encoded = np.array([
           -0.45467922-0.8906553j , -0.51913923+0.8546897j ,
            0.99873143-0.05035446j, -0.48378995+0.8751841j ,
            0.8763384 +0.48169592j, -0.8554102 -0.5179511j ,
            0.50102246-0.8654343j , -0.50232255+0.8646803j ,
            0.876649  +0.48113045j, -0.8551498 -0.51838094j,
            0.5562552 +0.83101153j,  0.51993096-0.8542083j ,
           -0.99772054-0.06748131j,  0.48131755-0.87654626j,
            0.49023774-0.87158877j], dtype=np.complex64)
    elif encoding=='BLOSUM62':
        alphabet = np.array([
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
            'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X', '*'], dtype='<U1')
        encoded = np.array([
           -0.26695704-0.9637084j ,  0.61654115-0.7873227j ,
            0.89596575-0.44412315j,  0.9572207 -0.28935888j,
           -0.8933483 -0.4493649j ,  0.69961226-0.7145227j ,
            0.8500247 -0.5267428j ,  0.9853714 +0.17042053j,
            0.5077042 -0.86153144j, -0.93440825-0.35620385j,
           -0.93607223-0.35180783j,  0.6905609 -0.72327423j,
           -0.8275205 -0.5614355j , -0.99233466-0.12357972j,
            0.5013154 -0.8652646j ,  0.5061711 -0.8624331j ,
           -0.13414967-0.9909611j , -0.7313355 +0.68201786j,
           -0.9242165 -0.38186896j, -0.8691504 -0.49454784j,
            0.94218355-0.33509728j,  0.81015223-0.58621955j,
           -0.04461522-0.99900424j,  0.16116107+0.9869281j ], dtype=np.complex64)
    elif encoding=='PROPS':
        alphabet= np.array([
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
            'F', 'P', 'S', 'T', 'W', 'Y', 'V'], dtype='<U1')
        encoded = np.array([
           -0.85851073-0.51279557j,  0.21812595+0.9759206j ,
           -0.00420049+0.9999912j , -0.28015   +0.9599562j ,
           -0.65689373-0.7539832j ,  0.47489336+0.8800433j ,
           -0.01283448+0.9999176j , -0.978518  -0.20616122j,
            0.29292223+0.9561363j ,  0.07060943-0.99750406j,
            0.04576658-0.99895215j, -0.03847059+0.9992597j ,
            0.24712168-0.9689845j ,  0.40146118-0.91587603j,
           -0.7614468 +0.64822745j, -0.98953974+0.14426042j,
           -0.99279606+0.11981639j,  0.85201293-0.52352077j,
            0.99998915+0.00465518j, -0.18931031-0.9819173j ], dtype=complex64)
    codex = defaultdict(lambda: defval, zip(alphabet, encoded))
    return codex

def codexmap(data, codex):
    """Transform iterable using codex dictionary and return array."""
    if isinstance(codex, str):
        codex = get_codex(codex)
    uniq, inv = np.unique(data, return_inverse=True)
    return np.array([codex[u] for u in uniq])[inv].reshape(data.shape)

def seqtorecs(seq, start=0):
    end = start + seq.size
    locs = np.arange(start, end)
    dtype = np.dtype([
        ('loc', int), ('seq', (np.str_, 1)),
        ('enc', np.complex64), ('phase', bool)])
    seq = codexmap(seq, 'NA')
    revcomp = codexmap(seq[::-1], 'COMPL')
    recs = [np.empty(seq.size, dtype), np.empty(seq.size, dtype)]
    recs[0]['seq'] = seq
    recs[1]['seq'] = revcomp
    recs[0]['loc'] = locs
    recs[1]['loc'] = locs[::-1]
    recs[0]['phase'] = True
    recs[1]['phase'] = False
    return recs

def encode(meta, structs, encodings='SQUARE'):
    """Apply complex encoding to sequences."""
    alphabets = meta['alphabet'].repeat(2)
    encodings = np.atleast_1d(encodings)
    encodarr = []
    for encoding in encodings:
        if encoding in ('SQUARE', 'MAFFT', 'TRANS', 'AG', 'PUPY', 'NUC.4.4'):
            encodeds = []
            for struct in structs:
                encoded = codexmap(struct['seq'], encoding)
                copy = struct.copy()
                copy['enc'] = encoded
                encodeds.append(copy)
            encodarr.append(encodeds)
        elif encoding in ('BLOSUM62', 'PROPS'):
            encodeds = []
            for struct in structs:
                for frame in np.arange(3):
                    struct = struct[frame:]
                    struct = struct[:(struct.size//3)*3]
                    seq = np.ascontiguousarray(struct['seq'])
                    seq = seq.view(np.dtype((np.str_, 3)))
                    encoded = codexmap(seq, getcodons())
                    encoded = codexmap(encoded, get_codex('BLOSUM62'))
                    copy = struct.copy()
                    copy['enc'] = encoded.repeat(3)
                    encodeds.append(copy)
            encodarr.append(encodeds)
    return encodarr

def stft(encoded, window, detr='constant'):
    """Apply short-time fourier transform."""
    compl = np.iscomplexobj(encoded)
    if detr=='compcorr':
        detr = compcorr
    return signal.stft(encoded, 1, window, window.size, window.size-1, 
                      return_onesided=not compl, boundary=None, axis=0, detrend=detr)[2]

def get_window(window_type, window_size, one_sided=True):
    """Get fourier tranform window."""
    if one_sided:
        window_size *= 2
    if window_type=='welch':
        window = np.arange(1,window_size+1)
        window = 1 - ((window-(window_size+1)/2)/((window_size+1)/2))**2
    else:
        window = signal.get_window(window_type, window_size, False)
    if one_sided:
        window = window[window_size//2-1::-1]
    return window / window.sum()

def norm_freqs(freqs, norm_level=1):
    """Normalise signal frequencies."""
    freqs = freqs * np.abs(freqs)**(norm_level-1)
    norm = np.linalg.norm(freqs, axis=-1, keepdims=True)
    norm[norm==0] = 1
    freqs /= norm
    return freqs

def compcorr(encoded, axis=0):
    """Correct for GC composition."""
    encoded -= encoded.mean(axis, keepdims=True)
    encoded.real /= np.abs(encoded.real).mean(axis, keepdims=True)*2
    encoded.imag /= np.abs(encoded.imag).mean(axis, keepdims=True)*2
    return encoded

def freq_transform(encoded, encoding='SQUARE', window_size=210, window_type='boxcar',
                   detr='constant', norm_level=1, chunk_size=24):
    """Apply frequency-space transformation."""
    gap = 1
    weight = 1
    if encoding in ('BLOSUM62', 'PROPS'):
        gap = 3
        weight = 1/4
    dtype = np.dtype([
        ('loc', int), ('seq', (np.str_, 1)), ('encs', (np.complex64, window_size)),
        ('freqs', (np.complex64, window_size)), ('powers', (float, window_size)),
        ('chunks', (np.float32, (window_size//chunk_size)*(chunk_size-1))), ('cwt', (np.complex64, window_size)),
        ('phase', bool), ('weight', float)])
    window = get_window(window_type, window_size)
    valid_size = encoded.shape[0] - window_size - gap
    freqs = stft(encoded['enc'], window, detr).swapaxes(0,1)[gap:valid_size+gap]
    powers = np.abs(freqs)
    encs = np.lib.stride_tricks.as_strided(encoded['enc'], (valid_size+gap,window_size),
                                           (encoded['enc'].strides[0],)*2)[gap:]
    encs = compcorr(encs, -1)
    n_chunk = window_size // chunk_size
    chunks = encs.reshape(valid_size, n_chunk, chunk_size)
    encs *= window.reshape(1,-1)
    chunks = np.abs(fft.fft(chunks, axis=-1)[:,:,1:])
    chunks /= np.linalg.norm(chunks, axis=-1, ord=2, keepdims=True)
    chunks = chunks.reshape(-1, (window_size//chunk_size)*(chunk_size-1))
    struct = np.empty(valid_size, dtype)
    struct['loc'] = encoded['loc'][:valid_size]
    struct['seq'] = encoded['seq'][:valid_size]
    struct['encs'] = encs
    struct['freqs'] = freqs
    struct['powers'] = powers
    struct['chunks'] = chunks
    struct['phase'] = encoded['phase'][:valid_size]
    struct['weight'] = weight
    return struct

def prepdata(encodeds, encodings='SQUARE', window_size=210, chunk_size=24, window_type='boxcar', detr='constant', norm_level=1):
    """Apply encodings to sequence structs."""
    encodings = np.atleast_1d(encodings)
    data = [np.concatenate([freq_transform(enc, encoding, window_size, window_type, detr, norm_level, chunk_size) for enc in encod], 0)
            for encod, encoding in zip(encodeds, encodings)]
    return data

def get_data(datas, window_size=210, chunk_size=24, window_type='boxcar', encodings='SQUARE', detr='constant', norm_level=1):
    """Encode and normalise sequences, then pack into structs."""
    metalist, structlist, seqlist = [], [], []
    for data in datas:
        meta, seqs = data
        encods = encode(meta, seqs, encodings)
        structs = prepdata(encods, encodings, window_size, chunk_size, window_type, detr, norm_level)
        metalist.append(meta)
        structlist.append(structs)
        seqlist.append(seqs)
    return metalist, structlist, seqlist

def calcpm(seq1, seq2=None):
    """Caculate probability of bases matching due to chance."""
    if seq2 is None:
        seq1, seq2 = seq1[:,0], seq1[:,1]
    codex = {'A':0, 'T':1, 'U':1, 'C':2, 'G':3, 'Y':4, 'R':4, 'W':4,
             'S':4, 'K':4, 'M':4, 'D':4, 'V':4, 'H':4, 'B':4, 'X':4, 'N':4}
    probmatch = np.bincount(codexmap(seq1, codex), minlength=5) / seq1.size
    probmatch = probmatch * np.bincount(codexmap(seq2, codex), minlength=5) / seq2.size
    probmatch = probmatch.sum()
    probmatch
    return probmatch

def prepsearch(struct, guide_size=24, query=False):
    """Prepare search windows."""
    window_size = struct['encs'].shape[1]
    freqs = struct['encs'][:,:guide_size]
    freqs /= np.linalg.norm(freqs, axis=-1, ord=2, keepdims=True)
    freqs = np.concatenate((freqs.real, freqs.imag), axis=-1)
    powers = struct['chunks'][:,guide_size:]
    powers /= np.linalg.norm(powers, axis=-1, ord=2, keepdims=True)
    stack = np.concatenate((powers, freqs/3), axis=-1)
    stack /= np.linalg.norm(stack, axis=-1, ord=2, keepdims=True)
    return stack

def nnsearch(query, subject, guide_size=24, reduct='mip', index="Flat", n=1, sample='all'):
    """Run nearest neighbour search for a specified pair of sequences."""
    n = np.atleast_1d(n)
    dtype = np.dtype([
        ('locs', (int, 2)), ('seqs', ((np.str_, 1), 2)),
        ('score', float), ('phases', bool, 2), ('weight', float), ('query', float)])
    if reduct=='mip':
        sreduced = prepsearch(subject, guide_size)
        qreduced = prepsearch(query, guide_size, query=True)
    sampoints = np.arange(qreduced.shape[0])
    if isinstance(sample, float):
        if sample<1.0:
            sample = np.round(sample*qreduced.shape[0]).astype(int)
    if isinstance(sample, int):
        if sample<qreduced.shape[0]:
            sampoints = np.round(np.linspace(0, qreduced.shape[0]-1, sample, endpoint=True)).astype(int)
            qreduced = qreduced[sampoints]
    if isinstance(index, str):
        nn = faiss.index_factory(sreduced.shape[-1], index)
    else:
        nn = index
    if nn.is_trained == False:
        rng = default_rng()
        nn.train(rng.choice(sreduced, size=int(np.sqrt(sreduced.shape[0])), replace=False))
    nn.add(sreduced)
    _, matchlocs = nn.search(qreduced, int(n.max()))
    matchlocs = matchlocs[:,n-1].T.ravel()
    match = subject[matchlocs]
    query = np.ascontiguousarray(query[sampoints])
    query = np.lib.stride_tricks.as_strided(
        query, (n.size, query.size), (0, query.strides[0])).ravel()
    struct = np.empty(matchlocs.size, dtype)
    struct['locs'][:,0] = query['loc']
    struct['locs'][:,1] = match['loc']
    struct['seqs'][:,0] = query['seq']
    struct['seqs'][:,1] = match['seq']
    struct['score'] = np.real(query['freqs'] * match['freqs'].conj()).sum(1)
    struct['phases'] = np.stack([query['phase'], match['phase']], axis=1)
    struct['weight'] = (query['weight'] + match['weight'])/2
    
    order = np.lexsort([struct['score'], struct['locs'][:,0]])
    struct = struct[order]
    return struct

def nnsearches(queries, subjects, guide_size=24, reduct='mip', index="Flat", nmatch=1, sample='all'):
    """Run nearest-neighbour search for all sequence pairs."""
    structs = []
    for query, subject in zip(queries, subjects):
        structs.append(nnsearch(query, subject, guide_size, reduct, index, nmatch, sample))
    return structs

def multisearch(structlist, guide_size=24, reduct='mip', index="Flat", nmatch=1, sample='all'):
    """Run nearest-neighbour search."""
    nns = nnsearches(structlist[0], structlist[1], guide_size, reduct, index, nmatch, sample)
    for search in nns:
        search['query'] = True
    nns2 = nnsearches(structlist[1], structlist[0], guide_size, reduct, index, nmatch, sample)
    for search in nns2:
        search['locs'] = search['locs'][:,::-1]
        search['seqs'] = search['seqs'][:,::-1]
        search['phases'] = search['phases'][:,::-1]
        search['query'] = False
    nns.extend(nns2)
    nnarr = np.empty(len(nns), dtype=object)
    nnarr[:] = nns
    return nnarr

def get_windows(structs, seqlist, window_size):
    """Get window sequences."""
    phase1, phase2 = structs['phases'].T
    locs1, locs2 = structs['locs'].T
    seqs1 = np.stack([np.concatenate([seqs['seq'] for seqs in seqlist[0][::2]]),                         np.concatenate([seqs['seq'][::-1] for seqs in seqlist[0][1::2]])], axis=1)
    seqs2 = np.stack([np.concatenate([seqs['seq'] for seqs in seqlist[1][::2]]),                         np.concatenate([seqs['seq'][::-1] for seqs in seqlist[1][1::2]])], axis=1)

    winds1 = np.reshape(phase1*2-1, (-1,1)) * np.arange(1,window_size+1).reshape(1,-1)
    winds1 = locs1.reshape(-1,1) + winds1
    winds2 = np.reshape(phase2*2-1, (-1,1)) * np.arange(1,window_size+1).reshape(1,-1)
    winds2 = locs2.reshape(-1,1) + winds2
    seqs1 = np.squeeze(np.take_along_axis(seqs1[winds1], 1-1*phase1.reshape(-1,1,1), axis=2))
    seqs2 = np.squeeze(np.take_along_axis(seqs2[winds2], 1-1*phase2.reshape(-1,1,1), axis=2))
    return seqs1, seqs2

def filt_guides(seqs1, seqs2, order=None, max_pass=None, maxgaps=0, gapopen=0, match=1, gapext=0, mismatch=0, verbose=False):
    """Align and filter guide windows that have more than the specified number of gaps."""
    if order is not None:
        seqs1 = seqs1[order]
        seqs2 = seqs2[order]
    guide_size = seqs1.shape[1]
    seqs1 = seqs1.view(np.dtype((str, guide_size)))[:,0]
    seqs2 = seqs2.view(np.dtype((str, guide_size)))[:,0]
    submat = parasail.matrix_create("ATUCG", match, mismatch)
    sgfilt = np.zeros(seqs1.size, bool)
    
    if verbose:
        seqs1 = tqdm(seqs1)
    for i,(seq1,seq2) in enumerate(zip(seqs1,seqs2)):
        if max_pass is not None:
            if i==max_pass:
                break
        aln = parasail.sg_trace_scan_sat(seq1, seq2, gapopen, gapext, matrix=submat)
        query = np.array([aln.traceback.query]).view('U1')
        ref = np.array([aln.traceback.ref]).view('U1')
        ngaps = np.logical_or(query=='-',ref=='-').sum()
        sgfilt[i] = ngaps <= maxgaps
    if order is not None:
        order = np.argsort(order)
        sgfilt = sgfilt[order]
    return sgfilt

def alnseqs2(seqs1, seqs2, probmatch=0.25, gapopen=2, match=1, gapext=0, mismatch=0, verbose=False):
    """Align sequences."""
    window_size = seqs1.shape[1]
    seqs1 = seqs1.view(np.dtype((str, window_size)))[:,0]
    seqs2 = seqs2.view(np.dtype((str, window_size)))[:,0]
    submat = parasail.matrix_create("ATUCG", match, mismatch)
    totl = len(seqs1[0]) + len(seqs2[0])
    alns = ma.masked_array(np.full((seqs1.size,totl), np.uint8(2)),                        np.ones((seqs1.size,totl),bool), fill_value=np.uint8(2))
    
    if verbose:
        seqs1 = tqdm(seqs1)
    for i,(seq1,seq2) in enumerate(zip(seqs1,seqs2)):
        aln = parasail.sg_trace_scan_sat(seq1, seq2, gapopen, gapext, matrix=submat)
        query = np.array([aln.traceback.query]).view('U1')
        ref = np.array([aln.traceback.ref]).view('U1')
        gapq, gapr = query=='-', ref=='-'
        match = query==ref
        alns.data[i,:match.size] = match + gapq*2 + gapr*3
        alns.mask[i,:match.size] = False
    return alns


def stabil_binom(vals):
    """Apply arcsin stabilisation to binomially distributed values."""
    vals = np.arcsin((2*vals-1)/1) / np.pi * 2
    return vals

def destabil_binom(vals):
    """Inverse arcsin stabilisation."""
    vals = (np.sin(vals/2*np.pi)*1 + 1) / 2
    return vals

def calc_occs2(obs, width=None):
    """Count the numbers of k-mers."""
    length = obs.shape[0]
    if width is None:
        width = obs.max()+1
    occarr = obs + (np.arange(length)*width).reshape(-1,1)
    occarr = np.bincount(occarr.ravel(), minlength=length*width)
    occarr = occarr.reshape(length,width)
    return occarr

def kmer_occs(obs, mask=None, ksize=1, width=3):
    """Calculate the occurrence of alignment k-mers."""
    obs = as_strided(obs, (obs.shape[0],obs.shape[1]-ksize+1,ksize),                           (obs.strides[0],obs.strides[1],obs.strides[1]))
    obs = np.sum(obs * (obs.max()+1)**np.arange(ksize).reshape(1,1,-1), -1)
    if mask is not None:
        mask = as_strided(mask, (mask.shape[0],mask.shape[1]-ksize+1,ksize),                               (mask.strides[0],mask.strides[1],mask.strides[1]))
        mask = mask.any(-1)
        obs = (obs+1)*np.logical_not(mask)
    occarr = calc_occs2(obs, width+1)[:,1:]
    return occarr, obs

def calc_pindels(obs):
    """Calculate the corrected proportion of gaps in an alignment."""
    isgap = obs>=3
    isnuc = np.logical_or(obs==1, obs==2)
    isend = obs==0
    first = isgap[:,0]
    last = obs.shape[1] - np.argmin(isend[:,::-1],1) - 1
    last = isgap[np.arange(last.size),last]
    nindels = np.sum(np.logical_and(np.logical_not(isgap[:,:-1]), isgap[:,1:]), 1)
    nindels += np.logical_and(last==False, first==True)
    nindels[np.logical_and(nindels==0,first==True)] = 1
    nnucs = np.logical_and(isnuc[:,1:],isnuc[:,:-1]).sum(1)
    rawpinds = nindels / np.clip(nnucs+nindels, np.finfo(float).resolution, None)
    rawpinds = np.clip(rawpinds, 0, 0.5)
    pindels = 1 - (2-1/(1-rawpinds))
    return pindels

@tf.function
def clust_dists(in_poccs, comps, samp_vars, binpow=8., eps=1e-6):
    """Calculate distance from each point to nearest cluster."""
    diffs = tf.expand_dims(in_poccs, -3) - tf.expand_dims(comps, -2)
    dists = tf.sqrt(tf.reduce_sum(diffs**2/samp_vars[:,None,:], -1))
    simils = 1/tf.maximum(dists, eps)
    simils = simils/tf.norm(simils, axis=-1, keepdims=True)
    simils = simils**2
    clustw = simils*2 - 1
    clustw = tf.abs(clustw)**(1/binpow) * tf.sign(clustw)
    clustw = clustw/2 + 0.5
    return dists, simils, clustw

@tf.function
def score_poccs(in_poccs, comps, samp_vars, stabil_pm, learn_mult=1., binpow=8., eps=1e-6):
    """Calculate loss for current parameters."""
    in_poccs = 1/(1+tf.exp(-in_poccs))
    in_poccs = in_poccs*2 - 1
    stabil_pm = tf.reshape(stabil_pm, (1,)*len(in_poccs.shape))
    in_poccs = tf.concat([in_poccs, stabil_pm], -1)
    shape = tf.concat([in_poccs.shape[:-1], [in_poccs.shape[-1]//comps.shape[-1],comps.shape[-1]]], 0)
    in_poccs = tf.reshape(in_poccs, shape)
    
    dists, simils, clustw = clust_dists(in_poccs, comps, samp_vars, binpow, eps)
    loss = tf.reduce_sum(tf.reduce_sum(dists**2*clustw, -2) / (tf.reduce_sum(clustw, -2)-1))
    loss *= learn_mult
    return loss, in_poccs, dists, clustw

def calc_elliptw(xs, ys):
    """Apply elliptical distance metric from each point to cluster centers."""
    clustw = np.sqrt(xs**4 + 2*xs**2*(ys**2-1) + (ys**2+1)**2)
    clustw = np.sqrt(xs**2 + ys**2 + 1 - clustw)
    clustw = np.sign(xs) * clustw / np.sqrt(2)
    return clustw

def grad_desc2(comps, mean_preds, samp_vars, matches, pindels, probmatch=0.25, ndists=1, weights=None,                binpow=16, reltol=1e-6, maxiter=100, learn_mult=0.01, prior_size=0, eps=1e-15, verbose=False):
    """Apply gradient descent algorithm to find homologous and non-homologous cluster centers."""
    test_poccs = comps[mean_preds>probmatch][np.argmin(np.abs(mean_preds[mean_preds>probmatch]-np.quantile(mean_preds[mean_preds>probmatch], 0.9)))]
    null_poccs = comps[np.argmin(np.abs(mean_preds-probmatch))]
    in_poccs = np.concatenate([test_poccs, null_poccs[:-1]])
    in_poccs = (np.clip(in_poccs, eps-1, 1-eps)+1)/2
    in_poccs = np.log(in_poccs/(1-in_poccs))
    
    stabil_pm = stabil_binom(probmatch)
    params = [comps, samp_vars, stabil_pm, learn_mult, binpow, eps]
    params = [np.float32(p) for p in params]
    func = lambda in_poccs: score_poccs(in_poccs, *params)[0]
    val_grad = lambda in_poccs: tfp.math.value_and_gradient(func, in_poccs)
    opt = tfp.optimizer.bfgs_minimize(
        val_grad,
        initial_position=tf.constant(in_poccs, tf.float32),
        tolerance=0,
        f_relative_tolerance=reltol,
        max_iterations=maxiter)
    
    pos = opt.position
    loss, out_poccs, cdists, clustw = score_poccs(pos, *params)
    loss, out_poccs, cdists, clustw = loss.numpy(), out_poccs.numpy(), cdists.numpy(), clustw.numpy()
    if verbose:
        print('niters: ' + str(opt.num_iterations.numpy()))
        print('loss: ' + str(opt.objective_value.numpy()))
        print('grad: ' + str(opt.objective_gradient.numpy()))
    
    diff = out_poccs[...,:-1,:] - out_poccs[...,-1:,:]
    bases = np.sqrt(np.sum(diff[None,:,:]**2/samp_vars[:,None,:], -1))
    sides = np.append(cdists, bases, -1)
    semipers = sides.sum(-1) / 2
    areas = np.sqrt(np.clip(semipers*np.prod(semipers[...,None]-sides, -1), 0, None))
    heights = 2 * areas[...,None] / bases
    projs = np.sqrt(cdists.max(-1,keepdims=True)**2-heights**2)
    mask = cdists.argmax(-1) != (cdists.shape[-1]-1)
    projs[mask] = bases[mask] - projs[mask]
    projs, heights = projs/bases*2-1, heights/bases
    clustw = calc_elliptw(projs, heights)
    clustw = clustw/2 + 0.5
    if binthres is None:
        clustw = clustw*2 - 1
        clustw = np.abs(clustw)**(1/binpow) * np.sign(clustw)
        clustw = clustw/2 + 0.5
    else:
        clustw = (clustw>binthres) * 1.
    clustw = np.append(clustw, np.max(1-clustw,-1,keepdims=True), -1)
    
    clust_sizes = np.clip(clustw.sum(0), 1, None)
    pids = []
    
    pid = None
    if out_poccs.shape[-1]>=2:
        out_poccs = destabil_binom(out_poccs)
        ngaps = out_poccs[...,-2] * clust_sizes * 1.5
        mtch = out_poccs[...,-1] * clust_sizes
        pid = (mtch+(prior_size-ngaps)*probmatch) / np.clip(clust_sizes+prior_size-ngaps, 1, None)
        pids.append(norm_pid(pid.max(), probmatch))
    return clustw, pids

def norm_pid(percid, probmatch, lower=0.25, clip=False):
    """Normalise proportion identity to range 0-1."""
    pid = (percid-probmatch)/(1-probmatch)
    if clip:
        pid = np.clip(pid, 0, 1)
    pid = pid*(1-lower) + lower
    return pid

def JCdist(norm, clip=10, limit=0.25):
    """Convert identity to substitution distance via Jukes-Cantor model."""
    factor = 1 - limit
    mindist = np.e**(-1/factor*clip)
    norm = np.clip(norm, mindist, 1)
    return np.abs(-factor*np.log(norm))

    """."""


# In[ ]:


with StringIO(infile1.read()) as f:
    genome1 = read_fasta(f)
with StringIO(infile2.read()) as f:
    genome2 = read_fasta(f)
infile1.close()
infile2.close()


# In[307]:


metalist, structlist, seqlist =     get_data([genome1, genome2], window_size, chunk_size, window_shape, encodings, detr, norm_level)


# In[308]:


seq1, seq2 = seqlist[0][0]['seq'], seqlist[1][0]['seq']
probmatch = calcpm(seq1, seq2)


# In[309]:


nns = multisearch(structlist, guide_size, reduct, index, nmatch, sample)
structs = np.concatenate(nns, 0)
_, mask = np.unique(structs, return_index=True)
structs = structs[mask]
coords = structs['locs']
seqs = structs['seqs']
matches = seqs[:,0]==seqs[:,1]
concord = structs['phases'][:,0]==structs['phases'][:,1]


# In[310]:


seqs1, seqs2 = get_windows(structs, seqlist, window_size)
guides1 = np.ascontiguousarray(seqs1[:,:filt_size])
guides2 =  np.ascontiguousarray(seqs2[:,:filt_size])
sgfilt = filt_guides(guides1, guides2, verbose=verbose)


# In[311]:


order = np.argsort(-structs['score'])
rev = np.argsort(order)
sgfilt = sgfilt[order]
opts = np.nonzero(sgfilt)[0]
if opts.size > max_pass:
    pass_filt = np.array([arr[0] for arr in np.array_split(opts, max_pass-1)])
    pass_filt = np.append(pass_filt, opts[-1])
    sgfilt[:] = False
    sgfilt[pass_filt] = True
sgfilt = sgfilt[rev]


# In[312]:


structs = structs[sgfilt]
coords = structs['locs']
seqs = structs['seqs']
probmatch = calcpm(seqs)
matches = seqs[:,0]==seqs[:,1]
concord = structs['phases'][:,0]==structs['phases'][:,1]


# In[313]:


seqs1, seqs2 = get_windows(structs, seqlist, window_size)
alns = alnseqs2(seqs1, seqs2, probmatch, 2, 1, verbose=verbose)
nsyms, sym_obs = kmer_occs(np.clip(alns.data, 0, 2), alns.mask, 1)


# In[314]:


shape = (sym_obs.shape[0], sym_obs.shape[1]-samp_width+1, samp_width)
strides = (sym_obs.strides[0], sym_obs.strides[1], sym_obs.strides[1])
samp_obs = as_strided(sym_obs, shape, strides)
samp_pinds = calc_pindels(samp_obs.reshape(-1,samp_width)).reshape(samp_obs.shape[:-1])
samp_occs = calc_occs2(samp_obs.reshape(-1,samp_width))[:,1:].reshape(samp_obs.shape[0], samp_obs.shape[1], -1)
samp_poccs = samp_occs/np.clip(samp_occs.sum(-1,keepdims=True), 1, None)
samp_pmatchs = samp_poccs[...,1]


# In[315]:


samp_vals = samp_pmatchs[...,np.newaxis]
samp_vals = stabil_binom(samp_vals)
samp_diffs = np.sqrt(np.sum((samp_vals-samp_vals[:,:1,:])**2*samp_width, -1))
samp_diffs[np.any(samp_obs==0, axis=-1)] = np.inf
samp_cuts = np.argmax(samp_diffs>cut_thres, 1) - 1
samp_cuts = np.clip(samp_cuts-samp_width+1, 2, None)


# In[316]:


cutfilt = samp_cuts>=min_samps
structs = structs[cutfilt]
coords = structs['locs']
seqs = structs['seqs']
probmatch = calcpm(seqs)
matches = seqs[:,0]==seqs[:,1]
concord = structs['phases'][:,0]==structs['phases'][:,1]
alns = alns[cutfilt]
samp_vars = (samp_obs, samp_pinds, samp_pmatchs, samp_diffs, samp_cuts)
samp_obs, samp_pinds, samp_pmatchs, samp_diffs, samp_cuts = [samp_var[cutfilt] for samp_var in samp_vars]


# In[317]:


samp_matchs = as_strided(matches, (samp_obs.shape[0],samp_obs.shape[1]), (matches.strides[0],0))
samp_cids = np.arange(samp_obs.shape[0])
samp_cids = as_strided(samp_cids, (samp_obs.shape[0],samp_obs.shape[1]), (samp_cids.strides[0],0))
samp_mask = np.zeros((samp_obs.shape[0], samp_obs.shape[1]), bool)
samp_mask[np.arange(samp_cuts.size), samp_cuts] = True
samp_mask = np.maximum.accumulate(samp_mask[:,::-1], 1)[:,::-1]
samp_mask *= np.all(samp_obs!=0, axis=-1)


# In[318]:


samp_inv = np.stack(np.nonzero(samp_mask), 1)
samp_near = np.nonzero(samp_inv[:,1]==0)[0]
samp_obs, samp_matchs, samp_cids, samp_pinds, samp_pmatchs, samp_diffs =     samp_obs[samp_mask], samp_matchs[samp_mask], samp_cids[samp_mask],     samp_pinds[samp_mask], samp_pmatchs[samp_mask], samp_diffs[samp_mask]


# In[319]:


alns.mask = np.concatenate([np.full((samp_mask.shape[0],samp_width-1),False), np.logical_not(samp_mask)], -1)
nsyms, sym_obs = kmer_occs(np.clip(alns.data, 0, 2), alns.mask, 1)
nmismats, nmatchs, ngaps = nsyms.T
pmatchs = nmatchs / nsyms.sum(1)
pmismats = nmismats / nsyms.sum(1)
pgaps = ngaps / nsyms.sum(1)
pindels = calc_pindels(sym_obs)


# In[320]:


vals = np.stack([samp_pmatchs, samp_pinds], 1)
reg = lgb.LGBMRegressor(num_leaves=nleaves, max_depth=-1, mc=(1,-1), mc_method='advanced',                         n_jobs=ncpu, n_estimators=ntrees, learning_rate=learn_rate,                         subsample=subsamp, subsample_freq=1*(subsamp<1))
reg.fit(vals, samp_matchs.ravel())
samp_preds = reg.predict(vals)
samp_preds = np.clip(samp_preds, 0, 1)


# In[321]:


predarr = np.full(samp_mask.shape, -1.)
predarr[samp_inv[:,0], samp_inv[:,1]] = stabil_binom(samp_preds)
mean_preds = np.average(predarr, weights=samp_mask, axis=1)
std_preds = np.sqrt(np.average((predarr-mean_preds[:,np.newaxis])**2, weights=samp_mask, axis=1))
predarr = destabil_binom(predarr)
mean_preds = destabil_binom(mean_preds)


# In[322]:


comps = np.stack([samp_pmatchs, samp_pinds, samp_preds], 1)
comps = stabil_binom(comps)
samp_vars = np.stack([stats.binned_statistic(samp_cids, comps[:,c], 'std', np.arange(matches.size+1)).statistic**2                       for c in np.arange(comps.shape[1])], -1)
comps = np.stack([samp_pmatchs, samp_pinds, np.clip(samp_preds, probmatch, None)], 1)
comps = stabil_binom(comps)
nobs = samp_mask.sum(1)
minvar = np.finfo(np.float32).resolution
samp_vars = np.clip(samp_vars, minvar, None)
samp_vars = samp_vars[samp_cids]


# In[337]:


pred_mask = np.ones(samp_preds.shape, bool)
comps = comps[pred_mask]
samp_vars = samp_vars[pred_mask]

clustw, pids = grad_desc2(comps, samp_preds[pred_mask], samp_vars, matches, pindels, probmatch=probmatch, ndists=1, weights=None,                binpow=binpow, reltol=reltol, maxiter=maxiter, learn_mult=learn_mult, prior_size=prior_size, eps=eps, verbose=verbose)
clustw = np.stack([stats.binned_statistic(samp_cids[pred_mask], clustw[:,c], 'mean', np.arange(matches.size+1)).statistic                for c in np.arange(clustw.shape[-1])], -1)
clust_sizes = clustw.sum(0)


# In[338]:


mtch = matches.reshape(-1,1)
pgs = pindels.reshape(-1,1)
ngaps = np.sum(mtch*pgs, 0)
pid = (mtch.sum(0)+(prior_size-ngaps)*probmatch) / np.clip(clust_sizes+prior_size-ngaps, 1, None)
pids.append(norm_pid(pid.max(), probmatch))

mtch = mean_preds.reshape(-1,1) * clustw
pgs = pindels.reshape(-1,1) * clustw
ngaps = np.sum(mtch*pgs, 0)
pid = (mtch.sum(0)+(prior_size-ngaps)*probmatch) / np.clip(clust_sizes+prior_size-ngaps, 1, None)
pids.append(norm_pid(pid.max(), probmatch))

mtch = matches.reshape(-1,1) * clustw
pgs = pindels.reshape(-1,1) * clustw
ngaps = np.sum(mtch*pgs, 0)
pid = (mtch.sum(0)+(prior_size-ngaps)*probmatch) / np.clip(clust_sizes+prior_size-ngaps, 1, None)
pids.append(norm_pid(pid.max(), probmatch))

pids = np.array(pids)
dists = JCdist(norm_pid(pids, 0.25, 0, True))


# In[339]:


out_dist = np.maximum(dists[1], dists[-1])


# In[ ]:


outfile.write(str(out_dist))
outfile.close()

