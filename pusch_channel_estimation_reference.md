# PUSCH Channel Estimation — Technical Reference

**Source tree:** `cuPHY/src/cuphy/ch_est/`  
**MATLAB reference:** `5GModel/nr_matlab/pxsch/`  
**Spec reference:** 3GPP TS 38.211 §6.4.1.1 (DMRS), TS 38.214 §6.2 (PUSCH)

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [DMRS Structure](#2-dmrs-structure)
3. [Algorithm Variants](#3-algorithm-variants)
4. [Common Preprocessing — Descrambling, tOCC, fOCC](#4-common-preprocessing)
5. [Algorithm 0: LEGACY_MMSE](#5-algorithm-0-legacy_mmse)
6. [Algorithm 1: MULTISTAGE_MMSE_WITH_DELAY_EST](#6-algorithm-1-multistage_mmse_with_delay_est)
7. [Algorithm 2: RKHS](#7-algorithm-2-rkhs)
8. [Algorithm 3: LS_ONLY](#8-algorithm-3-ls_only)
9. [MMSE Filter Coefficients](#9-mmse-filter-coefficients)
10. [Delay Estimation](#10-delay-estimation)
11. [Shift and Un-Shift Sequences](#11-shift-and-un-shift-sequences)
12. [DFT-s-OFDM DMRS Support](#12-dft-s-ofdm-dmrs-support)
13. [Per-PRG Channel Estimation](#13-per-prg-channel-estimation)
14. [CUDA Parallelism and Memory Model](#14-cuda-parallelism-and-memory-model)
15. [Key Data Structures](#15-key-data-structures)
16. [Tensor Layouts](#16-tensor-layouts)
17. [Heterogeneous UE-Group Processing](#17-heterogeneous-ue-group-processing)
18. [Execution Modes: Graph vs. Stream](#18-execution-modes-graph-vs-stream)
19. [ML / TensorRT Path](#19-ml--tensorrt-path)
20. [Constants and Compile-Time Parameters](#20-constants-and-compile-time-parameters)
21. [End-to-End Data Flow](#21-end-to-end-data-flow)
22. [Noise Power (N0) Estimation and SNR Treatment](#22-noise-power-n0-estimation-and-snr-treatment)
23. [Multiple Time Instances (chEstTimeInst)](#23-multiple-time-instances-chestimeinst)

---

## 1. Purpose and Scope

PUSCH channel estimation produces the complex channel matrix

```
H[rxAnt, layer, subcarrier, timeInst]
```

from the Demodulation Reference Signals (DMRS) embedded in the received PUSCH slot.
This estimate is consumed downstream by the MMSE equalizer and SINR measurement blocks.

The implementation lives in `ch_est/ch_est.cu` (≈6,835 lines) and is instantiated
at compile time across every combination of:

| Template axis | Values |
|---|---|
| `TStorage` | `float`, `__half` |
| `TDataRx` | `float`, `__half` |
| `TCompute` | `float` |
| `N_DMRS_GRIDS_PER_PRB` | 2 (Type 1), 3 (Type 2) |
| `N_DMRS_PRB_IN_PER_CLUSTER` | 4, 8 |
| `N_DMRS_INTERP_PRB_OUT_PER_CLUSTER` | 2, 4 |
| `N_DMRS_SYMS` | 1, 2 |

---

## 2. DMRS Structure

### 2.1 Type 1 Grid (N_DMRS_GRIDS_PER_PRB = 2)

DMRS tones occupy every other subcarrier within a PRB:

```
PRB subcarriers:  0  1  2  3  4  5  6  7  8  9  10  11
Grid 0 tones:     D     D     D     D     D      D
Grid 1 tones:        D     D     D     D     D       D
```

- 6 DMRS tones per grid per PRB (`N_DMRS_GRID_TONES_PER_PRB = 12/2 = 6`)
- Inter-grid frequency shift: **2 subcarriers**
- Supports up to 4 CDM groups → 4 layers

### 2.2 Type 2 Grid (N_DMRS_GRIDS_PER_PRB = 3)

DMRS tones occupy every third subcarrier group:

```
PRB subcarriers:  0  1  2  3  4  5  6  7  8  9  10  11
Grid 0 tones:     D  D           D  D           D   D
Grid 1 tones:           D  D           D  D
Grid 2 tones:                 D  D           D  D
```

- 4 DMRS tones per grid per PRB (`N_DMRS_GRID_TONES_PER_PRB = 12/3 = 4`)
- Inter-grid frequency shift: **1 subcarrier**
- Supports up to 6 layers

### 2.3 tOCC (Temporal Orthogonal Cover Code)

Layers are multiplexed in time using a [+1,+1] / [+1,−1] cover code over `dmrsMaxLen` consecutive DMRS symbols:

| tOCC index | Symbol weights |
|---|---|
| 0 | `[+1, +1]` → sum |
| 1 | `[+1, −1]` → difference |

`dmrsMaxLen = 1`: single-symbol DMRS, only tOCC 0 is active.  
`dmrsMaxLen = 2`: double-symbol DMRS, both tOCCs active → 2× layer capacity.

### 2.4 fOCC (Frequency Orthogonal Cover Code)

Within a time slot, layers that occupy the same DMRS grid are separated by a [+1,+1] / [+1,−1] frequency cover code applied to alternating tones:

| fOCC index | Tone weights |
|---|---|
| 0 | all `+1` |
| 1 | even `+1`, odd `−1` |

The `activeTOCCBmsk` and `activeFOCCBmsk` bitmasks in `cuphyPuschRxUeGrpPrms_t` select which OCC combinations are active for a given UE group.

### 2.5 Layer-to-OCC-Grid Mapping

Each layer is identified by a packed byte `OCCIdx[layer]`:

```
bits [1:0]  = fOCC index within the tOCC slot
bit  [2]    = DMRS grid index (0 or 1 for Type 1)
```

This encodes the (grid, tOCC, fOCC) triple that a layer occupies.

---

## 3. Algorithm Variants

Selected at runtime via `cuphyPuschChEstAlgoType_t`:

```c
typedef enum {
    PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE                    = 0,
    PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST = 1,
    PUSCH_CH_EST_ALGO_TYPE_RKHS                           = 2,
    PUSCH_CH_EST_ALGO_TYPE_LS_ONLY                        = 3,
} cuphyPuschChEstAlgoType_t;
```

| Algorithm | Delay Est. | MMSE Filter | Latency | Quality |
|---|---|---|---|---|
| LEGACY_MMSE | static (precomputed) | offline W | lowest | good |
| MULTISTAGE_MMSE | online (R1) | adaptive W | medium | better |
| RKHS | implicit (eigenspace) | RKHS | highest | best |
| LS_ONLY | none | none | lowest | basic |

---

## 4. Common Preprocessing

All algorithms share the initial preprocessing steps:
descrambling, tOCC removal, and fOCC removal.
These are implemented in both `windowedChEstNoDftSOfdmKernel` (legacy, all-in-one) and
`windowedChEstPreNoDftSOfdmKernel` (multi-stage first kernel).

### 4.1 Load DMRS Tones from Global Memory

Each thread loads one received sample from the OFDM grid:

```
shPilots[smemToneIdx][symIdx][smemGridIdx] =
    tDataRx[absSubcarrierIdx][dmrsSymPos[symIdx]][bsAntIdx]
```

Reads are coalesced (preserving GMEM order) but writes are scattered into shared memory
to separate the two DMRS grids:

```
SMEM_DMRS_TONE_IDX = threadIdx / nDmrsGridsPerPrb    (tone within grid)
SMEM_DMRS_GRID_IDX = threadIdx % nDmrsGridsPerPrb    (grid index)
```

### 4.2 Descrambling Sequence Generation

Per 3GPP TS 38.211 §6.4.1.1.1.1, the Gold-code initialization is:

```
cInit = 2^17 × (slotNum×14 + symIdx + 1) × (2×dmrsScrmId + 1)
      + 2×dmrsScrmId + scid
cInit &= ~(1<<31)          // clear bit 31
```

32 descrambler bits are generated at a time via `gold32(cInit, bitOffset)`.
Each tone needs 2 bits (1 for I, 1 for Q). Multiple threads collaborate to populate
`descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS]` in shared memory:

```
DMRS_DESCR_SEQ_WR_WORD_IDX = threadIdx % N_DMRS_DESCR_WORDS
DMRS_DESCR_SEQ_WR_SYM_IDX  = threadIdx / N_DMRS_DESCR_WORDS
```

### 4.3 Descrambling Application

The descrambling code for each tone is extracted from `descrWords`:

```
descrIBit = (descrWords[symIdx][wordIdx] >> bitIdx) & 0x1
descrQBit = (descrWords[symIdx][wordIdx] >> (bitIdx+1)) & 0x1

descrCode = conj( (1-2×descrIBit)/√2  +  j×(1-2×descrQBit)/√2 )
```

This produces one of four QPSK descrambling values: `{±1/√2 ± j/√2}`.

The shift sequence (for delay centering) is applied simultaneously:

```
descrShiftSeq = shiftSeq[toneIdx] × descrCode
shPilots[tone][sym][grid] ×= descrShiftSeq
```

### 4.4 tOCC Removal

For each active tOCC:

```
// tOCC index 0:  [+1, +1] cover code → coherent sum
avg[0] = (pilots[0] + pilots[1]) × (1/N_DMRS_SYMS)

// tOCC index 1:  [+1, -1] cover code → difference
avg[1] = (pilots[0] - pilots[1]) × (1/N_DMRS_SYMS)
```

For `N_DMRS_SYMS = 1`, no averaging is possible:

```
avg[0] = pilots[0]
```

### 4.5 fOCC Removal

For each active fOCC × tOCC combination:

```
sign = 1 if (fOCCIdx == 0) or (toneIdx is even)
     = -1 if (fOCCIdx == 1) and (toneIdx is odd)

shH[tone][N_DMRS_SYMS_FOCC×toccIdx + foccIdx][grid] = avg[toccIdx] × sign
```

After fOCC removal, `shH[tone][layer][grid]` contains per-layer **LS channel estimates**
at DMRS tone positions only (6 tones per PRB for Type 1).

---

## 5. Algorithm 0: LEGACY_MMSE

**Kernel:** `windowedChEstNoDftSOfdmKernel`  
**Source:** `ch_est.cu` lines 424–1069

This is a single monolithic kernel that performs all steps in one pass.
The shift sequence is **precomputed offline** based on a known (or fixed) delay assumption.

### 5.1 Kernel Launch Configuration

```
gridDim  = (nPrbClusters, nRxAnts, nUeGroups)
blockDim = (N_PRB_IN_PER_CLUSTER × 12, 1, 1)
```

where `nPrbClusters = ceil(nPrb / N_DMRS_INTERP_PRB_OUT_PER_CLUSTER)`.

Each thread block processes:
- One PRB cluster (4 or 8 input PRBs → 4 output PRBs)
- One Rx antenna

### 5.2 PRB Cluster Indexing

The MMSE filter operates on a sliding window of `N_DMRS_PRB_IN_PER_CLUSTER` PRBs
to estimate `N_DMRS_INTERP_PRB_OUT_PER_CLUSTER` output PRBs:

```
N_EDGE_PRB = (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2

prbClusterStart = (clusterIdx × outputPrbsPerCluster) - N_EDGE_PRB
// Clamped for first and last clusters:
if clusterIdx == 0:          prbClusterStart = 0
if clusterIdx == last:       prbClusterStart = nPrb - inputPrbsPerCluster
```

The edge PRBs are loaded but their output is discarded — they exist only to give the
filter sufficient context at the band boundaries.

### 5.3 MMSE Frequency Interpolation

After shared memory contains the per-layer LS estimates, each thread computes one
interpolated output tone via a dot product against one row of the precomputed
MMSE filter **W**:

```
H_est[n_out] = Σ_{k=0}^{N_TOTAL_DMRS_GRID_TONES-1}
                   W[n_out + gridShiftIdx, k, filtIdx] × shH[k, layerIdx, gridIdx]
```

- `gridShiftIdx` offsets the filter row to handle the frequency shift between DMRS grids.
- `filtIdx` selects one of three filter variants:

| `filtIdx` | Context | Constant |
|---|---|---|
| 0 | Middle PRB clusters | `MIDDLE_INTERP_FILT_IDX` |
| 1 | Lower edge (first cluster) | `LOWER_EDGE_INTERP_FILT_IDX` |
| 2 | Upper edge (last cluster) | `UPPER_EDGE_INTERP_FILT_IDX` |

The filter tensor dimensions:

```
WFreq      : (N_INTERP_TONES + shift, N_DMRS_TONES, 3)  — for nPrb >= 8 && nPrb%4==0
WFreq4     : (25, 24, 3)                                 — for other nPrb
WFreqSmall : (37, 18, 3)                                 — for nPrb <= 3
```

### 5.4 Un-Shift and Write

```
H_est[n_out] ×= unShiftSeq[clusterToneIdx + gridShiftIdx]
```

If `nDmrsCdmGrpsNoData == 1` (only one CDM group carries data, so DMRS power is halved):

```
H_est[n_out] ×= √2
```

Result is written to `tHEst[bsAntIdx, layerIdx, absSubcarrierIdx, timeInst]`.

---

## 6. Algorithm 1: MULTISTAGE_MMSE_WITH_DELAY_EST

**Kernels:** Three-kernel pipeline  
**Source:** `ch_est.cu` lines 1810–1800 (pre), 1241 (reduction), 1280 (filter)

### 6.1 Stage 1 — LS Estimation + Delay Accumulation

**Kernel:** `windowedChEstPreNoDftSOfdmKernel`

Performs all preprocessing steps (§4) and then:

**LS estimate write:**
```
// Per-layer LS estimate stored for later MMSE filtering
tInfoDmrsLSEst[dmrsToneIdx/2, layerIdx, bsAntIdx, timeInst] =
    shH[DMRS_TONE_IDX, j, DMRS_GRID_IDX]
```

`tInfoDmrsLSEst` stores LS estimates only at DMRS positions (every other subcarrier
for Type 1), indexed by DMRS tone number (not subcarrier number).

**R1 accumulation for delay estimation:**

For single-layer:
```
R1_thread += conj(shH[tone, j, grid]) × shH[tone+1, j, grid]
```

For multi-layer (uses tone pairs to average out layer cross-terms):
```
avg0 = (shH[tone, j, grid] + shH[tone+1, j, grid]) / 2
avg1 = (shH[tone+2, j, grid] + shH[tone+3, j, grid]) / 2
R1_thread += conj(avg0) × avg1
```

Warp-level reduction using cooperative groups:
```cpp
R1_block.x = reduce(tile, R1_thread.x, plus<float>());
R1_block.y = reduce(tile, R1_thread.y, plus<float>());
```

Block-level accumulation to global memory via atomic adds:
```cpp
atomicAdd(&pAccum[activeBuffer].x, sh_R1[0].x);
atomicAdd(&pAccum[activeBuffer].y, sh_R1[0].y);
```

The inactive buffer is simultaneously zeroed by block (0,0):
```cpp
pAccum[activeBuffer ^ 1] = {0.0f, 0.0f};   // ping-pong clear
```

### 6.2 Stage 2 — Delay Mean Extraction

**Function:** `chEstDelayShiftReduction` (called from dispatch kernel, line 1241)

Only thread 0 of the first block (blockIdx.x=0, blockIdx.y=0) executes this:

```cpp
// Read accumulated R1 from global memory
TComplexCompute accum = pAccum[dmrsActiveAccumBuf];

// P_dmrs is the inter-tone spacing of the correlation:
//   1 layer  → adjacent DMRS tones are 2 subcarriers apart → P_dmrs = 2
//   >1 layer → tone pairs are 4 subcarriers apart          → P_dmrs = 4
float P_dmrs = (nLayers == 1) ? 2.0f : 4.0f;

// Delay mean in seconds
float delay_mean_sec = -atan2f(accum.y, accum.x) / (2π × P_dmrs);

// Delay mean in samples (normalize by subcarrier spacing)
sh_delay_mean = delay_mean_sec / df;   // df = 15e3 × 2^mu Hz
```

The full MATLAB equivalent from `pusch_ChEst_LS_delayEst_MMSE.m`:
```matlab
delay_mean = -angle(R1) / (2*pi * P_dmrs);    % in seconds
delay_mean_microsec = delay_mean / delta_f * 1e6;
```

All threads in the block read `sh_delay_mean` after `__syncthreads()`.

### 6.3 Stage 3 — Delay-Adaptive MMSE Filtering

**Kernel:** `chEstFilterNoDftSOfdmDispatchKernel` (line 1280)

Launch configuration:
```
gridDim  = (nPrbClusters, ceil(nRxAnts/antBlockDim), nUeGroups)
blockDim = (nLayers × N_INTERP_TONES_PER_CLUSTER, antBlockDim, 1)
```

Threads are partitioned: `LAYER_IDX = threadIdx.x / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER`.

**Step A — Generate shift and un-shift sequences in registers:**

```cpp
float partial_arg = df × 2π × sh_delay_mean;

// Shift sequence: applied to DMRS tones (stride 2)
shiftSeq[n].x = cosf(partial_arg × 2n);
shiftSeq[n].y = sinf(partial_arg × 2n);

// Un-shift sequence: applied to data subcarriers
unShiftSeq[n].x = cosf(-partial_arg × n);
unShiftSeq[n].y = sinf(-partial_arg × n);
```

**Step B — Load LS estimates from global memory into shared memory:**

```cpp
sh_ls_est[layer × N_DMRS_TONES + tone] =
    tInfoDmrsLSEst[toneOffset + tone, layer, bsAntIdx, timeInst] × shiftSeq[tone]
```

**Step C — MMSE filter (same dot product as Legacy path):**

```cpp
TComplexCompute accum = {0, 0};
for (int j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; j++) {
    accum += sh_ls_est[layer × N_DMRS_TONES + j] × W[interpToneIdx+gridShift, j, filtIdx];
}
```

**Step D — Un-shift and write:**

```cpp
accum ×= unShiftSeq[clusterToneIdx + gridShiftIdx] × scaling;
tHEst[bsAntIdx, layerIdx, absSubcarrierIdx, timeInst] = accum;
```

### 6.4 Dispatch Logic for Cluster Sizes

The filter kernel dispatches at runtime to compile-time-specialized versions:

```cpp
if (nPrb <= 3):             PRB_IN = nPrb,  PRB_OUT = nPrb
elif (nPrb >= 8 && nPrb%4 == 0):  PRB_IN = 8, PRB_OUT = 4
else:                       PRB_IN = 4,  PRB_OUT = 2
```

For the per-PRG variant (`chEstFilterPrgNoDftSOfdmDispatchKernel`), when `prgSize < 4`
each PRG maps directly to one cluster; when `prgSize == 4` clusters are 2-PRB output.

---

## 7. Algorithm 2: RKHS

**Kernel:** `puschRkhsChEstKernel`  
**Source:** `ch_est.cu` line 4387  
**Constants defined at top of file:**

```cpp
#define RKHS_MAX_N_GNB_ANTS       4
#define RKHS_N_DMRS_GRID_TONES_PER_PRB  6
#define RKHS_N_EIGS               3   // eigenvectors per PRB
#define RKHS_N_ZP_EIGS            6   // eigenvectors in zero-padded space
#define RKHS_MAX_LAYERS           4
#define RKHS_MAX_N_INTS           50  // max CP intervals
#define RKHS_USE_HAMMING              // Hamming windowing enabled
```

### 7.1 Mathematical Basis

In the frequency domain, the channel across the DMRS tones of one PRB can be approximated as lying in a low-dimensional subspace. The RKHS algorithm precomputes a basis (eigenvectors) that spans the typical channel space for a given number of PRBs.

Given a channel vector **h** (length `nDmrsSc = 6×nPrb`) at the DMRS tones, the representation is:

```
h ≈ V × c
```

where **V** is the `nDmrsSc × N_EIGS` eigenvector matrix and **c** is the `N_EIGS × 1` coefficient vector.

The received signal at DMRS tones is:

```
y = diag(h) × d + n   →   d^H × y ≈ (d^H × diag(d)) × V × c + noise
```

where **d** is the known DMRS pilot sequence.

The RKHS estimator finds the MAP solution for **c** given a prior derived from the
CP-delay-spread distribution.

### 7.2 Zero-Padding and Eigenvector Tables

To support varying `nPrb` values without storing a separate basis for each, the implementation
uses a **zero-padded (ZP) eigenvector** approach. A fixed-size ZP basis of `nZpDmrsSc`
tones is computed, and PRB-specific eigenvectors are obtained by:

```
eigVec[eig, tone] = EigVecCob[eig, :] × ZpDmrsScEigenVec[:, tone]
```

where `EigVecCob` (N_EIGS × N_ZP_EIGS) is the change-of-basis to collapse the ZP basis
to the PRB-specific basis.

`zpIdx` (from `prbRkhsDesc.zpIdx`) selects which ZP table to use, trading off memory
vs. precision for different PRB counts.

### 7.3 Kernel Launch Configuration

```
gridDim  = (nComputeBlocks, 1, 1)
blockDim = (nZpDmrsSc, 1, 1)          // one thread per zero-padded DMRS subcarrier
```

Each thread block handles one compute block (a contiguous set of PRBs for one UE group).
The compute block parameters are in `rkhsComputeBlockPrms_t`.

### 7.4 Per-Thread Eigenvector Projection

```cpp
// Compute eigenvector values for this thread's DMRS subcarrier
realMtrxVecMult<N_EIGS, N_ZP_EIGS>(
    eigVecValuesForThread,   // output: N_EIGS values
    tEigVecCob,              // N_EIGS × N_ZP_EIGS real matrix
    tZpDmrsScEigenVec.trace(THREAD_IDX)  // N_ZP_EIGS values for this subcarrier
);
```

### 7.5 Hamming Windowing

Applied to reduce spectral leakage before delay-domain noise estimation:

```cpp
float hammingPhase = 2π × THREAD_IDX / (nDmrsSc - 1);
hammingWindow = 0.54 - 0.46 × cos(hammingPhase);
```

### 7.6 DMRS Load and OCC Removal

For each DMRS grid (0 or 1), tOCC index, and Rx antenna:

```cpp
uint16_t scIdx = 2×THREAD_IDX + startSc + gridIdx;   // DMRS subcarrier (Type 1)
__half2 antRxData = tDataRx(scIdx, dmrsSymPos[0], antIdx) × descrCode[0];

if (dmrsMaxLen > 1) {
    // Second symbol: tOCC sign applied
    __half2 sym1 = tDataRx(scIdx, dmrsSymPos[1], antIdx);
    if (toccIdx == 1) sym1 = -sym1;
    antRxData = (antRxData + descrCode[1] × sym1) × RECIPROCAL_SQRT2;
}
```

### 7.7 Noise Estimation

RKHS is the **only algorithm with online noise power (N0) estimation**. Three methods are
supported, selected via `computeBlocksCommonPrms.noiseEstMethod`:

#### Method 0 — `USE_EMPTY_DMRS_GRID`

Used when `nDmrsCdmGrpsNoData == 1` (one CDM group carries data, leaving one empty).
The second DMRS grid carries no data pilot, so its received energy is pure noise:

```cpp
// gridBitMask selects only the active grid; the else branch handles the empty one
} else if ((noiseEstMethod == USE_EMPTY_DMRS_GRID) && (THREAD_IDX < nDmrsSc)) {
    for (int antIdx = 0; antIdx < nRxAnt; ++antIdx) {
        uint16_t scIdx = 2 * THREAD_IDX + startSc + gridIdx;
        for (int j = 0; j < dmrsMaxLen; ++j) {
            uint32_t symIdx = pDmrsSymPos[j];
            __half2 scRx = tDataRx(scIdx, symIdx, antIdx);
            noiseEnergyMeasuredByThread_half += scRx.x*scRx.x + scRx.y*scRx.y;
        }
    }
}
```

#### Method 1 — `USE_EMPTY_FOCC`

Used when a fOCC slot is inactive. The energy in the corresponding frequency-domain
bins (after the two-stage FFT of the eigenvector projection with `eigIdx == 0`) is
accumulated as noise:

```cpp
// Only for eigIdx == 0 projection, in bins corresponding to inactive fOCC
if (eigIdx == 0) {
    if (THREAD_IDX < nNoiseIntsPerGrid) {
        uint16_t corrIdx = foccIdx * nZpDmrsSc/2 + noiseRegionFirstIntIdx
                           + idxWithinNoiseRegion;
        __half2 corrValue = sh_fourierWorkspace[corrIdx];
        noiseEnergyMeasuredByThread_half += corrValue.x*corrValue.x
                                          + corrValue.y*corrValue.y;
    }
}
```

#### Method 2 — `USE_QUITE_FOCC_REGIONS` (Hamming path, default)

The received signal is **Hamming-windowed** before a two-stage FFT to the delay domain.
Delay-domain bins beyond the expected channel support are quiet (no multipath energy)
and their energy is pure noise:

```cpp
__half2 hammingCorr = { antRxData.x * hammingWindow,
                        antRxData.y * hammingWindow };

twoStageFourierTransform<log2SecondStageFourierSize>(
    THREAD_IDX, hammingCorr,
    tSecondStageTwiddleFactors, pSecondStageFourierPerm,
    sh_fourierWorkspace
);

if (THREAD_IDX < nNoiseIntsPerGrid) {
    uint8_t  noiseRegionIdx       = THREAD_IDX / nNoiseIntsPerFocc;
    uint16_t idxWithinNoiseRegion = THREAD_IDX - noiseRegionIdx * nNoiseIntsPerFocc;
    uint8_t  foccIdx              = (noiseRegionIdx + 1) % 2;
    uint16_t corrIdx              = foccIdx * nZpDmrsSc/2
                                   + noiseRegionFirstIntIdx + idxWithinNoiseRegion;

    __half2 corrValue             = sh_fourierWorkspace[corrIdx];
    noiseEnergyMeasuredByThread_half += corrValue.x*corrValue.x
                                      + corrValue.y*corrValue.y;
}
```

The noise bins are at indices `foccIdx × nZpDmrsSc/2 + noiseRegionFirstIntIdx + [0..nNoiseIntsPerFocc)`,
covering both fOCC regions that flank the channel support.

#### 7.7.1 Warp-to-Block Noise Reduction

Each thread accumulates a partial noise energy in `noiseEnergyMeasuredByThread_half`.
A warp-level reduction sums these into a per-warp result, then an `atomicAdd` merges
all warps into the single `sh_noiseEnergy` shared-memory accumulator:

```cpp
// Warp-level tree reduction:
float noiseEnergyWarp = (float)noiseEnergyMeasuredByThread_half;
for (int reduceStage = 16; reduceStage > 0; reduceStage /= 2)
    noiseEnergyWarp += tile.shfl_down(noiseEnergyWarp, reduceStage);

// Lane 0 of each warp contributes to shared accumulator:
if (LANE_IDX == 0)
    atomicAdd(&sh_noiseEnergy, noiseEnergyWarp);
__syncthreads();
```

#### 7.7.2 Hamming Correction and N0 Computation

The Hamming window attenuates each sample by `w[n]`, which reduces the measured noise
energy by a factor of `Σ w[n]² / N`. A linear correction restores the true noise level:

```cpp
float noiseScaling = 1.0f;   // default (Methods 0 and 1)
#ifdef RKHS_USE_HAMMING
if (noiseEstMethod > 1) {
    constexpr float a = 0.3974f;   // empirical slope
    constexpr float b = 0.0032f;   // empirical intercept
    noiseScaling = a * (float)nDmrsSc + b;
}
#endif

N0 = (__half)(sh_noiseEnergy / (noiseScaling * nNoiseMeasurments));
```

The scaling `a × nDmrsSc + b` is a linear fit across all supported PRB counts.
`nNoiseMeasurments` is the total number of noise bins accumulated
(set in `computeBlocksCommonPrms.nNoiseMeasurments`).

The final N0 (in linear scale, per subcarrier) is stored in `__half` precision and
used directly in the matching pursuit MMSE step.

### 7.8 Projection Coefficient Computation

The two-stage FFT (`twoStageFourierTransform`) used in §7.7 is also the engine for
computing projection coefficients. For each eigenvector `eigIdx`, every thread computes
`antRxData × eigVecValuesForThread[eigIdx]` and the FFT reduces across the DMRS tones,
placing the correlation result in `sh_fourierWorkspace[corrIdx]`:

```cpp
for (int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx) {
    __half2 eigenVecCorr = { antRxData.x * eigVecValuesForThread[eigIdx],
                              antRxData.y * eigVecValuesForThread[eigIdx] };

    twoStageFourierTransform<log2SecondStageFourierSize>(
        THREAD_IDX, eigenVecCorr,
        tSecondStageTwiddleFactors, pSecondStageFourierPerm,
        sh_fourierWorkspace
    );

    // Threads active in matching pursuit (threadActiveInMp) read their CP interval slot:
    if (threadActiveInMp && projCoeffAntIdx == antIdx) {
        for (int foccIdx = 0; foccIdx < 2; ++foccIdx) {
            if (toccPrm.foccBitMask >> foccIdx) {
                uint16_t corrIdx     = projCoeffIntIdx + foccIdx * nZpDmrsSc / 2;
                uint8_t  layerIdx    = toccPrm.foccPrms[foccIdx].layerIdx;
                projCoeffs[layerIdx][eigIdx] = sh_fourierWorkspace[corrIdx];
            }
        }
    }
}
```

`projCoeffIntIdx = THREAD_IDX / nRxAnt` — each thread owns one CP interval for one antenna.
`projCoeffAntIdx = THREAD_IDX % nRxAnt`.

The correlation matrices `sh_corr[cpInt][eigIn][eigOut]` (used in §7.9) are preloaded into
shared memory from `tCorr` before the grid/tOCC/ant loops.

#### 7.8.1 antPSD (Signal Power Estimate per Layer)

Inside the matching pursuit loop, the signal power for each layer at the candidate
CP interval is estimated from the projection coefficients, weighted by the eigenvalues:

```cpp
for (int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx) {
    antPSD[ueLayerIdx] = 0;
    for (int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx) {
        uint8_t  layerIdx  = ueLayerIdx + ueLayerStartIdx;
        __half2  projCoeff = projCoeffs[layerIdx][eigIdx];
        antPSD[ueLayerIdx] += (tEigVal(eigIdx) / sumEigValues)
                              * (projCoeff.x*projCoeff.x + projCoeff.y*projCoeff.y);
    }
    // antCoeffEnergy: combined power across all layers at this CP interval/antenna
    antCoeffEnergy += antPSD[ueLayerIdx];

    // Subtract noise floor and normalize by zero-padded size:
    antPSD[ueLayerIdx] = (antPSD[ueLayerIdx] - N0) / (__half)nZpDmrsSc;
    if (antPSD[ueLayerIdx] < 0) antPSD[ueLayerIdx] = 0;   // floor at 0
}
```

`sumEigValues` is the sum of all `RKHS_N_EIGS` eigenvalues for this PRB count (stored in
`prbRkhsDesc.sumEigValues`). The ratio `tEigVal(eigIdx) / sumEigValues` weights each
eigenvector dimension by its relative importance in the channel covariance.

### 7.9 Matching Pursuit (Greedy CP Interval Selection)

The RKHS estimator builds the channel estimate by greedily selecting the `nCpInt`
most energetic CP intervals, one per matching pursuit iteration. Each iteration:

#### Step 1 — Reduce `antCoeffEnergy` to a single `coeffEnergy` across antennas

```cpp
// Sum antCoeffEnergy across the nRxAnt threads that share the same projCoeffIntIdx:
coeffEnergy = antCoeffEnergy;
for (int reduceStage = nRxAnt / 2; reduceStage > 0; reduceStage /= 2)
    coeffEnergy += tile.shfl_down(coeffEnergy, reduceStage);
```

#### Step 2 — Warp-level max search

```cpp
maxCoeffEnergyInWarp = coeffEnergy;
maxCoeffInWarpIdx    = projCoeffIntIdx;
for (int reduceStage = nRxAnt; reduceStage < 32; reduceStage *= 2) {
    __half   prop_energy = tile.shfl_down(maxCoeffEnergyInWarp, reduceStage);
    uint16_t prop_idx    = tile.shfl_down(maxCoeffInWarpIdx,    reduceStage);
    if (prop_energy > maxCoeffEnergyInWarp) {
        maxCoeffEnergyInWarp = prop_energy;
        maxCoeffInWarpIdx    = prop_idx;
    }
}
// Lane 0 writes to shared memory:
if (LANE_IDX == 0) {
    sh_warpMaxCoeffEnergy[WARP_IDX] = maxCoeffEnergyInWarp;
    sh_warpMaxCoeffIdx[WARP_IDX]    = maxCoeffInWarpIdx;
}
__syncthreads();
```

#### Step 3 — Block-level max search (serial, in registers)

```cpp
__half   maxCoeffEnergy = 0;
uint16_t maxCoeffIdx    = 0;
for (int warpIdx = 0; warpIdx < nWarpsActiveInMp; ++warpIdx) {
    if (sh_warpMaxCoeffEnergy[warpIdx] > maxCoeffEnergy) {
        maxCoeffEnergy = sh_warpMaxCoeffEnergy[warpIdx];
        maxCoeffIdx    = sh_warpMaxCoeffIdx[warpIdx];
    }
}
```

#### Step 4 — Exit criterion

```cpp
__half exitCriteria = (__half)3 * N0 * (__half)nRxAnt * (__half)nUeLayers[ueIdx];
if (maxCoeffEnergy < exitCriteria) break;
```

The loop terminates when the residual energy at the most energetic CP interval falls
below `3 × N0 × nRxAnt × nUeLayers`. This prevents fitting noise.

#### Step 5 — Compute `eqCoeff` with Wiener filter (lambda)

The thread that owns the winning CP interval (`maxCoeffIdx == projCoeffIntIdx`) computes
the MMSE regularized coefficient:

```cpp
if (maxCoeffIdx == projCoeffIntIdx) {
    updateFlag = 1;
    for (int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx) {
        for (int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx) {
            __half2 eqCoeff = projCoeffs[layerIdx][eigIdx];

            // Wiener regularization factor:
            __half lambda = (antPSD[ueLayerIdx] * tEigVal(eigIdx))
                          / (antPSD[ueLayerIdx] * tEigVal(eigIdx) + N0);
            eqCoeff.x = lambda * eqCoeff.x;
            eqCoeff.y = lambda * eqCoeff.y;

            sh_tEqCoeff(eigIdx, ueLayerIdx, projCoeffAntIdx, nEqCoeffs) = eqCoeff;
        }
    }
    sh_eqIntIdxs[nEqCoeffs] = projCoeffIntIdx;   // record which CP interval
}
```

`lambda = (signal_power × eigVal) / (signal_power × eigVal + N0)` is the standard
Wiener MMSE regularizer: it approaches 1 when signal dominates noise and 0 otherwise.

#### Step 6 — Update residual projection coefficients (`deltaProjCoeff`)

All active threads subtract the contribution of the selected CP interval from the
remaining projection coefficients, preventing double-counting in subsequent iterations:

```cpp
// Correlation between projCoeffIntIdx and maxCoeffIdx:
int distBoxIdxToUpdateBox  = abs(projCoeffIntIdx - maxCoeffIdx);
bool boxIdxLessTheUpdateBox = (projCoeffIntIdx < maxCoeffIdx);

// Load the cross-correlation matrix (conjugate for negative offset):
for (i, j) { corr[i][j] = tCorr(j, i, distBoxIdxToUpdateBox);
             if (boxIdxLessTheUpdateBox) corr[i][j].y = -corr[i][j].y; }

// Subtract: projCoeffs -= corr × newEqCoeff
cplxMtrxVecMult(deltaProjCoeff[ueLayerIdx], corr, newEqCoeff[ueLayerIdx]);
projCoeffs[ueLayerIdx][eigIdx] -= deltaProjCoeff[ueLayerIdx][eigIdx];
```

After this, `nEqCoeffs` is incremented. The loop repeats until the exit criterion fires
or `nCpInt` iterations complete. All selected CP interval indices are in `sh_eqIntIdxs[0..nEqCoeffs-1]`.

### 7.10 Interpolation to Data Subcarriers

After matching pursuit, each thread (with `THREAD_IDX < nOutputSc`) computes the
channel estimate for its assigned output subcarrier by summing contributions from all
`nEqCoeffs` selected CP intervals:

#### Step 1 — Load interpolation eigenvectors for this subcarrier

```cpp
__half interpVecValuesForThread[RKHS_N_EIGS];
realMtrxVecMult<RKHS_N_EIGS, RKHS_N_ZP_EIGS>(
    interpVecValuesForThread,
    tInterpCob,                       // N_INTERP_EIGS × N_ZP_EIGS change-of-basis
    tZpInterpVec.trace(scIdx)         // ZP interpolation eigenvectors for this sc
);
```

#### Step 2 — Per-eqIdx accumulation with delay phase (`wave`)

For each selected CP interval `eqIdx`:

```cpp
int eqIntIdx = sh_eqIntIdxs[eqIdx];   // CP delay-domain index

// Compute the phasor for this subcarrier and CP interval (per DMRS grid):
for (int gridOffset = 0; gridOffset < 2; ++gridOffset) {
    float phase = -π × (float)(scIdx - gridOffset) × (float)eqIntIdx
                      / (float)nZpDmrsSc;
    __sincosf(phase, &wave[gridOffset].y, &wave[gridOffset].x);
}
```

The `wave` phasor shifts the estimate in delay by `eqIntIdx` CP bins, accounting for
the inter-grid offset `gridOffset` (0 for grid 0, 1 for grid 1 in Type 1 DMRS).

#### Step 3 — Compute interpolated channel and apply CDM scaling

```cpp
for (int eqIdx = 0; eqIdx < nEqCoeffs; ++eqIdx) {
    for (int ueLayerIdx ...) {
        uint8_t gridIdx = gridIdxs[layerIdx];   // DMRS grid for this layer
        for (int antIdx ...) {
            // Inner product of equalization coefficients with interpolation basis:
            __half2 interpValue = {0,0};
            for (int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx) {
                __half2 eqCoeff = sh_tEqCoeff(eigIdx, ueLayerIdx, antIdx, eqIdx);
                interpValue.x  += interpVecValuesForThread[eigIdx] * eqCoeff.x;
                interpValue.y  += interpVecValuesForThread[eigIdx] * eqCoeff.y;
            }

            // Apply delay-domain phasor:
            modulatedInterpValue = complex_mul(wave[gridIdx], interpValue);

            // CDM scaling: when both CDM groups carry data, DMRS power per grid is
            // halved (√2 attenuation), so the channel estimate must be corrected:
            if (drvdUeGrpPrm.nDmrsCdmGrpsNoData == 2) {
                modulatedInterpValue.x *= RECIPROCAL_SQRT2;  // ×(1/√2)
                modulatedInterpValue.y *= RECIPROCAL_SQRT2;
            }

            H_est_sc[ueLayerIdx][antIdx] += modulatedInterpValue;
        }
    }
}
```

#### Step 4 — Write to `tHEst`

```cpp
tHEst(antIdx, layerIdx, THREAD_IDX + scOffsetIntoChEstBuff, chEstTimeInst).x
    = (float)H_est_sc[ueLayerIdx][antIdx].x;
tHEst(antIdx, layerIdx, THREAD_IDX + scOffsetIntoChEstBuff, chEstTimeInst).y
    = (float)H_est_sc[ueLayerIdx][antIdx].y;
```

`scOffsetIntoChEstBuff` is the absolute subcarrier offset of the compute block's output
region within the full channel estimate buffer.

---

## 8. Algorithm 3: LS_ONLY

**Kernel:** Same as MULTISTAGE_MMSE Stage 1, but the filter stage is skipped.

The LS estimates in `tInfoDmrsLSEst` are written directly to `tHEst` without any
MMSE smoothing. Only descrambling, tOCC/fOCC removal are applied.

This is useful as a performance baseline or for high-SNR scenarios where the
channel is sparse and noise is negligible.

---

## 9. MMSE Filter Coefficients

**Source:** `derive_chest_mmse_coeff.m` and precomputed offline, stored in static tensors.

### 9.1 Filter Derivation Model

The MMSE Wiener filter is derived from two assumptions:

**Noise model:** AWGN with SNR = 10⁻³ (−30 dB). This is deliberately conservative —
it ensures the filter never over-smooths and adapts well across a wide SNR range.

**Channel model:** Uniform Power Delay Profile (PDP) of duration:

```
mmse_rect_pdp_len = √12 × delay_spread
```

(The factor √12 converts from RMS delay spread to the equivalent rectangular window length.)

The Wiener filter in the frequency domain for a rectangular PDP of length `τ_max`
and subcarrier spacing `Δf` is:

```
R_hh[k] = sinc(k × Δf × τ_max)   // channel autocorrelation
W = R_hh × (R_hh + SNR^{-1} × I)^{-1}
```

The matrix **W** maps `N_DMRS_TONES` LS estimates to `N_INTERP_TONES` channel estimates.
For Type 1 DMRS with 8 input PRBs and 4 output PRBs, this is a `(48×2) × (48)` matrix,
i.e., `96 × 48`.

### 9.2 Three Filter Banks

Three filter variants are precomputed to handle the non-stationary boundary conditions
at the edges of the allocated bandwidth:

| Variant | Position | Behavior |
|---|---|---|
| `filtIdx=0` (middle) | Interior clusters | Symmetric FIR, both sides equal |
| `filtIdx=1` (lower edge) | First cluster | One-sided, no context below |
| `filtIdx=2` (upper edge) | Last cluster | One-sided, no context above |

### 9.3 Filter Tensor Variants

Three tensor sizes handle different allocation ranges:

| Tensor | Use case | Input | Output |
|---|---|---|---|
| `WFreq` | `nPrb >= 8 && nPrb%4 == 0` | 8 PRBs (48 tones/grid) | 4 PRBs |
| `WFreq4` | Other `nPrb >= 4` | 4 PRBs (24 tones/grid) | 2 PRBs |
| `WFreqSmall` | `nPrb <= 3` | up to 3 PRBs (18 tones/grid) | same |

---

## 10. Delay Estimation

### 10.1 R1 Statistic

The delay mean is estimated from the phase of the lag-1 frequency autocorrelation R1:

```
R1 = Σ_k  H_LS*(k) × H_LS(k + P_dmrs)
```

where the sum is over all DMRS tone pairs that are `P_dmrs` subcarriers apart.

For Type 1 DMRS:
- **1 layer:** Adjacent DMRS tones are 2 subcarriers apart → `P_dmrs = 2`
- **>1 layer:** To average out the inter-layer interference, pairs of tone averages
  separated by 4 subcarriers are used → `P_dmrs = 4`

The phase of R1 is linearly proportional to the mean delay:

```
∠R1 = 2π × delay_mean × Δf × P_dmrs
→ delay_mean = -∠R1 / (2π × Δf × P_dmrs)   [seconds]
```

### 10.2 Double-Buffer Accumulation

Two accumulation buffers are used ping-pong style (`dmrsActiveAccumBuf ∈ {0,1}`):

- The **active buffer** accumulates R1 contributions across all PRB clusters and antennas
  via `atomicAdd`.
- The **inactive buffer** is cleared to zero within the same kernel invocation (by block 0).
- The active buffer index is toggled between time instances to avoid race conditions with
  overlapping kernel launches.

### 10.3 Per-PRG Delay Estimation

When `enablePerPrgChEst = 1`, R1 accumulation excludes certain tone pairs to prevent
inter-PRG boundary contamination:

```cpp
// For prgSize=1: exclude tone at position 4 within each 6-tone grid segment
if (prgSize == 1 && DMRS_TONE_IDX % 6 == 4) isValidR1Cal = false;
if (prgSize == 2 && DMRS_TONE_IDX % 12 == 10) isValidR1Cal = false;
if (prgSize == 3 && DMRS_TONE_IDX % 18 == 16) isValidR1Cal = false;
if (prgSize == 4 && DMRS_TONE_IDX % 24 == 22) isValidR1Cal = false;
```

---

## 11. Shift and Un-Shift Sequences

These sequences implement delay-domain centering: they shift the channel impulse response
to zero delay before MMSE filtering (which assumes zero-centered channel), then shift it
back.

### 11.1 Shift Sequence (applied to DMRS tones)

DMRS tones are at even subcarriers (stride 2 for Type 1). The `n`-th DMRS tone maps to
subcarrier `2n`, so:

```
shiftSeq[n] = exp(+j × 2π × delay_mean × Δf × 2n)
```

In the CUDA kernel (`chEstFilterNoDftSOfdmDispatchKernel`, line 1334):

```cpp
float partial_arg = df × 2π × sh_delay_mean;
float f_dmrs = partial_arg × (2 × toneIdx);
sincosf(f_dmrs, &sh_shiftSeq[i].y, &sh_shiftSeq[i].x);
```

### 11.2 Un-Shift Sequence (applied to all data subcarriers)

Data subcarriers are at positions `0, 1, 2, ..., nPrb×12−1`. The un-shift corrects the
effect of the frequency-domain shift:

```
unShiftSeq[n] = exp(-j × 2π × delay_mean × Δf × n)
```

Note the sign flip and the different index origin:

```cpp
// index offset of -1 (f_data starts from -1) to handle the cluster boundary offset
float f_data = -1.0f × partial_arg × (float(i) - 1.0f);
sincosf(f_data, &sh_unShiftSeq[i].y, &sh_unShiftSeq[i].x);
```

### 11.3 Legacy (Static) vs. Adaptive Sequences

| Path | Shift/Unshift source |
|---|---|
| LEGACY_MMSE | Static tensors precomputed from a fixed delay, stored in `statDescr.tPrmShiftSeq` / `tPrmUnShiftSeq` |
| MULTISTAGE_MMSE | Generated per-slot on GPU from estimated `delay_mean` |

For the legacy path, the MATLAB derivation is in `derive_chest_mmse_coeff.m`:

```matlab
df = 15e3 × 2^mu;
f_dmrs = df × (0:2:8×12-1)';          % DMRS subcarrier frequencies
shiftSeq = exp(+2π×j×delay_mean×f_dmrs);
f_data = df × (-1:8×12-1)';            % data subcarrier frequencies  
unShiftSeq = exp(-2π×j×delay_mean×f_data);
```

---

## 12. DFT-s-OFDM DMRS Support

When `enableDftSOfdm = 1` and `enableTfPrcd = 1` for a UE group, DMRS sequences
follow the DFT-s-OFDM waveform and use computer-generated sequences based on
Zadoff-Chu roots.

### 12.1 Sequence Generation

**Source:** `gen_pusch_dftsofdm_descrcode()`, line 303.

For `M_ZC < 36` (short sequences), hard-coded QPSK tables are used:

```cpp
// From phi_6, phi_12, phi_18, phi_24 tables (defined in constant memory)
descrCode.x = cos(π × phi[u][rIdx] / 4.0f);
descrCode.y = sin(π × phi[u][rIdx] / 4.0f);
```

For `M_ZC ≥ 36` (long sequences), a Zadoff-Chu sequence is used:

```
M_ZC = N_DMRS_GRID_TONES_PER_PRB × nPrb   // = 6 × nPrb
N = largest prime ≤ M_ZC

q̄ = N × (u+1) / 31
q = floor(q̄ + 0.5) + v × (sign)   // v is the sequence index

r(m) = exp(-jπ × q × m × (m+1) / N),   m = rIdx mod N
```

The sequence group `u` (0–29) is derived from the slot number and physical cell ID.

### 12.2 Prime Number Lookup

To avoid a runtime search for the largest prime ≤ `M_ZC`, a precomputed table is used:

```cpp
// d_primeNums[nPrb-1] gives the largest prime <= 6*nPrb
static __device__ __constant__ uint16_t d_primeNums[273];
```

Entries cover `nPrb` from 1 to 273 (maximum supported bandwidth).

---

## 13. Per-PRG Channel Estimation

**Enabled when:** `enablePerPrgChEst = 1` and `prgSize ∈ {1, 2, 3, 4}`

### 13.1 Motivation

Standard channel estimation uses a single delay estimate and filter for the entire
UE allocation. When the BS uses per-PRG precoding (different precoder per PRG),
the effective channel can vary across PRGs. Per-PRG channel estimation runs a
separate MMSE filter per PRG to track these variations.

### 13.2 PRG Cluster Mapping

For `prgSize < 4`, each PRG maps one-to-one to a filter cluster:

```
nPrbClusters = ceil(nPrb / prgSize)
clusterIdx → startPrb = clusterIdx × prgSize
```

For `prgSize = 4`, clusters are sized 2 output PRBs with 4-PRB input windows:

```
nPrbClusters = 2 × floor(nPrb/4) + (1 if remainder > 0)
clusterIdx 0,1 → startPrb = 0 (first 4 PRBs, two output clusters)
clusterIdx 2,3 → startPrb = 4 (next 4 PRBs, two output clusters)
...
```

This is implemented in `chEstFilterPrgNoDftSOfdmDispatchKernel`.

### 13.3 Edge Handling

Each PRG cluster acts as its own "band edge" and uses the lower-edge filter for
the first half and upper-edge filter for the second half:

```cpp
filtIdx = (PRB_CLUSTER_IDX % 2 == 0) ? LOWER_EDGE_INTERP_FILT_IDX
                                      : UPPER_EDGE_INTERP_FILT_IDX;
```

---

## 14. CUDA Parallelism and Memory Model

### 14.1 Thread Block Mapping

For `windowedChEstNoDftSOfdmKernel` (Legacy):

```
gridDim.x  = nPrbClusters           (one per PRB cluster)
gridDim.y  = nRxAnts                (one per Rx antenna)
gridDim.z  = nUeGroupsInHetCfg      (one per UE group slot in het. config)
blockDim.x = N_PRB_IN_PER_CLUSTER × 12   (one per tone in cluster)
```

For `chEstFilterNoDftSOfdmDispatchKernel` (Multi-stage filter):

```
gridDim.x  = nPrbClusters
gridDim.y  = ceil(nRxAnts / antBlockDim)
gridDim.z  = nUeGroupsInHetCfg
blockDim.x = nLayers × N_INTERP_TONES_PER_CLUSTER
blockDim.y = antBlockDim            (multiple antennas per block)
```

For `puschRkhsChEstKernel` (RKHS):

```
gridDim.x  = nComputeBlocks         (one per RKHS compute block)
blockDim.x = nZpDmrsSc              (one per ZP DMRS subcarrier)
```

### 14.2 Shared Memory Layout

**Legacy kernel** — shared memory is overlaid in two phases:

```
Phase 1 (pilots):  smemBlk[N_DMRS_TONES × N_DMRS_SYMS]
    shPilots: block_3D[N_DMRS_TONES_PER_CLUSTER][N_DMRS_SYMS][N_DMRS_GRIDS]

Phase 2 (H):       same smemBlk (reused after __syncthreads)
    shH:      block_3D[N_INTERP_TONES_PER_CLUSTER][N_DMRS_SYMS_OCC][N_DMRS_GRIDS]
```

The overlay is safe because `N_SMEM_ELEMS = max(pilots_elems, H_elems)`.
An assertion verifies both fit: `static_assert(shPilots.num_elem() <= N_SMEM_ELEMS)`.

**Multi-stage filter kernel:**

```
sh_shiftSeq[MAX_PRB_PER_FILTER × (12/nGrids)]
sh_unShiftSeq[(MAX_PRB_PER_FILTER × 12) + 1]
sh_ls_est[nRxAntsPerBlock × nLayers × MAX_DMRS_TONES_PER_CLUSTER]  // dynamic
sh_delay_mean[1]
```

`sh_ls_est` is dynamically allocated (`extern __shared__`) to handle varying nLayers.

### 14.3 Synchronization Points

| Location | Sync type | Reason |
|---|---|---|
| After GMEM load | `__syncthreads()` | Descrambler words written by different threads |
| After OCC removal | `thisThrdBlk.sync()` | shPilots → shH overlay, tOCCIdx varies per thread |
| After MMSE | `thisThrdBlk.sync()` | shH written by thread A, read by thread B |
| Delay reduction | `__syncthreads()` | thread 0 writes sh_delay_mean, all threads read |

`thisThrdBlk.sync()` (cooperative groups) is used instead of `__syncthreads()` in the
OCC stages because different threads may take different code paths (conditional tOCCIdx),
and `__syncthreads()` requires all threads to converge.

### 14.4 Memory Access Patterns

| Access | Pattern | Notes |
|---|---|---|
| `tDataRx` load | Coalesced | Threads load consecutive subcarriers |
| `descrWords` write | Bank-conflict-free | Interleaved by word and symbol |
| `shPilots` write | Scattered | Re-ordered to separate DMRS grids |
| `tFreqInterpCoefs` load | Strided | Each thread accesses one row |
| `tHEst` write | Scattered | One write per layer per output tone |

---

## 15. Key Data Structures

### 15.1 `cuphyPuschRxUeGrpPrms_t` — Per-UE-Group Parameters

Key fields used by channel estimation:

| Field | Type | Description |
|---|---|---|
| `nPrb` | `uint16_t` | Number of allocated PRBs |
| `startPrb` | `uint16_t` | First allocated PRB index |
| `nRxAnt` | `uint16_t` | Number of Rx antennas |
| `nLayers` | `uint16_t` | Number of MIMO layers |
| `dmrsScrmId` | `uint16_t` | DMRS scrambling ID (from RRC) |
| `scid` | `uint8_t` | Scrambling ID selection (0 or 1) |
| `dmrsMaxLen` | `uint8_t` | Max DMRS symbol length (1 or 2) |
| `dmrsSymLoc[]` | `uint8_t[]` | Symbol positions of DMRS within slot |
| `activeDMRSGridBmsk` | `uint8_t` | Bitmask of active DMRS grids |
| `activeTOCCBmsk[]` | `uint8_t[]` | Active tOCC indices per grid |
| `activeFOCCBmsk[]` | `uint8_t[]` | Active fOCC indices per grid |
| `OCCIdx[]` | `uint8_t[]` | Layer-to-OCC-grid mapping |
| `nDmrsCdmGrpsNoData` | `uint8_t` | CDM groups not carrying data (1 or 2) |
| `slotNum` | `uint16_t` | Slot number within frame |
| `mu` | `uint8_t` | Numerology (subcarrier spacing = 15×2^mu kHz) |
| `prgSize` | `uint16_t` | PRG size in PRBs (for per-PRG ChEst) |
| `enablePerPrgChEst` | `uint8_t` | Enable per-PRG estimation |
| `tInfoDataRx` | `cuphyTensorInfo3_t` | Input OFDM grid |
| `tInfoHEst` | `cuphyTensorInfo4_t` | Output channel estimates |
| `tInfoDmrsLSEst` | `cuphyTensorInfo4_t` | Intermediate LS estimates |
| `tInfoDmrsAccum` | `cuphyTensorInfo1_t` | R1 accumulation buffer (double) |
| `tInfoDmrsDelayMean` | `cuphyTensorInfo1_t` | Delay mean output |
| `dmrsActiveAccumBuf` | `uint8_t` | Active ping-pong buffer index |

### 15.2 `puschRxChEstStatDescr_t` — Static Descriptor

Holds precomputed, slot-invariant data:

| Field | Description |
|---|---|
| `tPrmFreqInterpCoefs` | MMSE filter W for 8-PRB clusters |
| `tPrmFreqInterpCoefs4` | MMSE filter W for 4-PRB clusters |
| `tPrmFreqInterpCoefsSmall` | MMSE filter W for ≤3-PRB |
| `tPrmShiftSeq` / `tPrmShiftSeq4` | Static shift sequences (legacy only) |
| `tPrmUnShiftSeq` / `tPrmUnShiftSeq4` | Static un-shift sequences (legacy only) |
| `pSymbolRxStatus` | Bitmask of received symbols (for early termination) |
| `prbRkhsDescs[]` | RKHS eigenvector tables per nPrb (1..273) |
| `zpRkhsDescs[]` | Zero-padded eigenvector tables |

### 15.3 `puschRxChEstDynDescr_t` — Dynamic Descriptor

Updated per slot:

| Field | Description |
|---|---|
| `chEstTimeInst` | Which time instance is being processed (0..nTimeChEsts-1) |
| `dmrsSymPos[]` | Absolute symbol positions of DMRS in this slot |
| `pDrvdUeGrpPrms` | Pointer to array of `cuphyPuschRxUeGrpPrms_t` |
| `hetCfgUeGrpMap[]` | Maps blockIdx.z → UE group index |
| `rkhsUeGrpPrms[]` | RKHS-specific UE group parameters |
| `rkhsCompBlockPrms[]` | RKHS compute block assignments |

---

## 16. Tensor Layouts

All tensors are column-major (C-contiguous in rightmost dimension).

### 16.1 Input: `tInfoDataRx`

```
Shape:    (nSubcarriers, nSymbols, nRxAnts)
Strides:  [1, nSubcarriers, nSubcarriers×nSymbols]
Element:  complex<TDataRx>   (typically __half2 / half2)
```

### 16.2 Intermediate: `tInfoDmrsLSEst`

```
Shape:    (nDmrsTones/2, nLayers, nRxAnts, nTimeInst)
Strides:  [1, nDmrsTones/2, nDmrsTones/2×nLayers, ...]
Element:  complex<TCompute>  (float2)
Note:     Only DMRS subcarrier positions are stored (stride-2 compression)
```

### 16.3 Output: `tInfoHEst`

```
Shape:    (nRxAnts, nLayers, nSubcarriers, nTimeInst)
Strides:  [nLayers×nSc×nT, nSc×nT, nT, 1]
Element:  complex<TStorage>  (float2 or __half2)
```

### 16.4 Delay Accumulation: `tInfoDmrsAccum`

```
Shape:    (2,)        // double buffer: ping[0] and pong[1]
Element:  complex<TCompute>
```

### 16.5 MMSE Filter: `tPrmFreqInterpCoefs`

```
Shape:    (N_INTERP_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT,
           N_DMRS_TONES_PER_CLUSTER,
           3)
Strides:  [1, N_INTERP_TONES+shift, (N_INTERP_TONES+shift)×N_DMRS_TONES]
Element:  TCompute   (float)
Note:     The '+shift' in dim 0 accommodates the gridShiftIdx offset
```

---

## 17. Heterogeneous UE-Group Processing

Multiple UE groups with different configurations (PRB count, layer count, antenna count)
can be processed in a single kernel launch. This is called the "heterogeneous config" (het-cfg).

### 17.1 Het-Cfg Limits

```cpp
CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS      // per LEGACY_MMSE
CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS  // per MULTISTAGE
CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS = 16    // maximum
```

### 17.2 UE Group Dispatch

`blockIdx.z` is the "het-cfg slot index". The mapping to actual UE group index is:

```cpp
const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
```

UE groups that don't fill all `blockIdx.z` slots are handled by early exits:

```cpp
if (PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_FOR_THIS_UEG || BS_ANT_IDX >= nRxAnt) return;
```

---

## 18. Execution Modes: Graph vs. Stream

**Source:** `ch_est_graph_mgr.cpp` and `ch_est_stream.cpp`

| Mode | Class | Use case |
|---|---|---|
| CUDA Graph | `puschRxChEstGraphMgr` | Full-slot, minimum latency, precompiled kernel graph |
| Stream | `puschRxChEstStream` | Pipeline integration, more flexible ordering |

Both modes implement the `IGraph_mgr` / `IStream` interface and select the appropriate
kernels based on `chEstAlgo`, `nDmrsGridsPerPrb`, and compute type.

### 18.1 Kernel Selection Flow

```
puschRxChEstKernelBuilder::kernelSelectL2()
    ├── if RKHS:     rkhsKernelSelectL1()
    ├── if MULTISTAGE_MMSE:
    │       windowedChEstPreNoDftSOfdmKernel   (Stage 1)
    │       chEstFilterNoDftSOfdmDispatchKernel (Stage 3)
    │       or chEstFilterPrgNoDftSOfdmDispatchKernel (per-PRG)
    ├── if LEGACY_MMSE:
    │       windowedChEstNoDftSOfdmKernel       (DFT-s-OFDM disabled)
    │       or windowedChEstKernel              (DFT-s-OFDM enabled)
    └── if LS_ONLY:
            windowedChEstPreNoDftSOfdmKernel    (Stage 1 only, no filter)
```

---

## 19. ML / TensorRT Path

**Source:** `trtengine_chest.cpp`, `ch_est_trtengine_pre_post_conversion.cu`

A neural network can replace the classical MMSE filter (Stage 3). The ML model is
specified via `puschrxChestFactorySettingsFilename` YAML config.

### 19.1 Pre-processing: `prepareChestMlInputsKernel`

Converts LS estimates from complex to real-valued pairs:

```
Input:  tInfoDmrsLSEst (complex float, nDmrsTones × nLayers × nRxAnts)
Output: ML model input tensor (real float, 2 × nDmrsTones × nLayers × nRxAnts)
```

### 19.2 ML Inference

The TensorRT engine runs inference on the prepared real-valued LS estimates.

### 19.3 Post-processing: `extractChestMlOutputsKernel`

Converts the real-valued ML output back to complex channel estimates:

```
Input:  ML model output (real float)
Output: tHEst (complex float/half)
```

The ML path is selected via `chest_factory.cpp` when a TRT engine file is configured.

---

## 20. Constants and Compile-Time Parameters

| Constant | Value | Description |
|---|---|---|
| `N_TONES_PER_PRB` | 12 | Subcarriers per PRB |
| `N_TOCC` | 2 | Maximum tOCC codes |
| `N_DMRS_SYMS_OCC` | 4 | Max OCC combinations reserved in SMEM |
| `N_MAX_DMRS_SYMS` | 4 | Max DMRS symbols per slot |
| `CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST` | 4 | Max temporal channel estimates |
| `OFDM_SYMBOLS_PER_SLOT` | 14 | Symbols per slot (normal CP) |
| `RKHS_N_EIGS` | 3 | Eigenvectors per PRB |
| `RKHS_N_ZP_EIGS` | 6 | ZP-domain eigenvectors |
| `MAX_N_PRBS_SUPPORTED` | 273 | Max PRB count (3300 subcarriers / 12) |
| `NUM_RKHS_ZP` | varies | Number of ZP table entries |
| `MAX_N_USER_GROUPS_SUPPORTED` | varies | Max UE groups per slot |

---

## 21. End-to-End Data Flow

```
                        ┌─────────────────────────────────────────────┐
                        │         tDataRx [NF × ND × nRxAnt]         │
                        │         (received OFDM resource grid)        │
                        └──────────────────┬──────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────┐
                    │  Stage 1: windowedChEstPreNoDftSOfdmKernel       │
                    │  (MULTISTAGE / LS_ONLY)  OR                      │
                    │  windowedChEstNoDftSOfdmKernel (LEGACY)           │
                    │                                                   │
                    │  1. Load DMRS tones [coalesced GMEM read]         │
                    │  2. Gold-code descrambling (38.211 §6.4.1.1.1.1) │
                    │  3. Delay centering ×shiftSeq                     │
                    │  4. tOCC removal [+1,+1] / [+1,-1] averaging      │
                    │  5. fOCC removal [+1,+1] / [+1,-1] sign flip      │
                    │  6. Write tInfoDmrsLSEst                           │
                    │  7. Accumulate R1 → tInfoDmrsAccum (via atomicAdd)│
                    └──────────┬───────────────────────┬───────────────┘
                               │ MULTISTAGE path        │ LEGACY path
                    ┌──────────▼──────────┐            │
                    │ chEstDelayShift-    │            │ (uses static shiftSeq
                    │ Reduction (thread 0)│            │  and filter W)
                    │                     │            │
                    │ delay_mean =        │            │
                    │  -∠R1/(2π×P_dmrs)  │            │
                    └──────────┬──────────┘            │
                               │                        │
            ┌──────────────────▼──────────────────────▼──────────────────┐
            │  Stage 3: chEstFilterNoDftSOfdmDispatchKernel               │
            │  (or chEstFilterPrgNoDftSOfdmDispatchKernel for per-PRG)     │
            │                                                               │
            │  1. Generate shiftSeq[n]=exp(+j×2π×delay×Δf×2n) on-GPU      │
            │  2. Generate unShiftSeq[n]=exp(-j×2π×delay×Δf×n) on-GPU     │
            │  3. Load tInfoDmrsLSEst → apply shiftSeq → sh_ls_est         │
            │  4. MMSE interp: H = Σ_k W[out,k,filtIdx] × sh_ls_est[k]    │
            │     (3 filter banks: middle / lower-edge / upper-edge)        │
            │  5. H ×= unShiftSeq × CDM_scaling                            │
            │  6. Write tHEst [nRxAnt × nLayers × nSc × nTimeInst]         │
            └──────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┤  RKHS path (Algorithm 2)
                │              │
    ┌───────────▼──────────────┴──────────────────────────────────────────┐
    │  puschRkhsChEstKernel                                                │
    │                                                                       │
    │  1. Load ZP eigenvector + correlation tables into SMEM               │
    │  2. Descramble DMRS tones (Gold code, 38.211 §6.4.1.1.1.1)          │
    │  3. tOCC removal (+1/+1 or +1/-1 over dmrsMaxLen symbols)            │
    │  4. Noise estimation (one of three methods):                          │
    │     a. Empty DMRS grid  → direct |rx|² sum                           │
    │     b. Empty fOCC bins  → eigVec[0] FFT quiet bins                   │
    │     c. Hamming FFT      → delay-domain quiet regions (default)       │
    │     Warp reduce + atomicAdd → sh_noiseEnergy                         │
    │     N0 = sh_noiseEnergy / (noiseScaling × nNoiseMeasurments)         │
    │          noiseScaling = 0.3974×nDmrsSc + 0.0032 (Hamming path)      │
    │  5. Project onto eigenvectors (2-stage FFT per eigIdx):               │
    │     projCoeffs[layer][eig][antIdx][cpIntIdx]                          │
    │  6. Matching Pursuit (greedy, up to nCpInt iterations):               │
    │     a. antPSD = Σ_eig (eigVal/sumEigValues)×|projCoeff|² − N0       │
    │     b. coeffEnergy = Σ_ant antPSD; warp+block max search            │
    │     c. EXIT if maxCoeffEnergy < 3×N0×nRxAnt×nUeLayers               │
    │     d. lambda = antPSD×eigVal / (antPSD×eigVal + N0)  (Wiener)      │
    │     e. eqCoeff = lambda × projCoeff                                  │
    │     f. projCoeffs −= corr × eqCoeff  (orthogonality update)         │
    │     nEqCoeffs selected CP intervals stored in sh_eqIntIdxs[]         │
    │  7. Interpolation (per output sc, per eqIdx):                         │
    │     phase = −π×(scIdx−gridOffset)×eqIntIdx/nZpDmrsSc               │
    │     H += interpCob×ZpInterpVec × eqCoeff × exp(j×phase)              │
    │     CDM scale: nDmrsCdmGrpsNoData==2 → ×1/√2                        │
    │  8. Write tHEst[antIdx, layerIdx, sc+offset, chEstTimeInst]          │
    └──────────────────────────────────────────────────────────────────────┘
```

---

## 22. Noise Power (N0) Estimation and SNR Treatment

The four algorithms differ fundamentally in how they handle noise power:

### 22.1 LEGACY_MMSE — Static SNR Baked Offline

The MMSE filter **W** is precomputed offline using a fixed SNR = 10⁻³ (−30 dB):

```matlab
% From derive_chest_mmse_coeff.m:
SNR = 10^-3;
R_hh = sinc((0:N-1)' * df * mmse_rect_pdp_len);   % channel autocorrelation
W = R_hh * inv(R_hh + SNR^-1 * eye(N));
```

The filter coefficients in `WFreq`, `WFreq4`, `WFreqSmall` already embed the noise
regularization. There is no runtime noise measurement, no N0 variable, and no SNR
computation anywhere in the LEGACY kernel path. The "10⁻³" is a design choice:
it is conservative enough to avoid over-smoothing at low SNR while still providing
useful interpolation gain at high SNR.

### 22.2 MULTISTAGE_MMSE — Static SNR, Adaptive Delay

The MULTISTAGE path adds online delay estimation (§6) but the MMSE filter W is still
the same offline-baked `WFreq` tensor. The delay adaptation changes the shift/un-shift
sequences but does **not** update the noise regularization level. N0 is not computed
at runtime.

When `SimCtrl.alg.ChEst_enable_update_W = 1` (MATLAB reference only), the delay spread
is estimated online and a new W is derived:

```matlab
mmse_rect_pdp_len = sqrt(12) * delay_spread_microsec * 1e-6;
tmp_table = derive_chest_mmse_coeff(carrier.mu, delay_mean, mmse_rect_pdp_len);
```

In the production CUDA implementation, W is not updated per-slot — only the shift and
un-shift sequences change.

### 22.3 RKHS — Fully Online N0 via Matching Pursuit

RKHS is the **only path with a runtime noise power estimate**. N0 is computed per
compute block (a contiguous PRB range) per kernel invocation:

```
N0 = sh_noiseEnergy / (noiseScaling × nNoiseMeasurments)
```

N0 appears in two places:

1. **Matching pursuit exit criterion** (§7.9):
   ```
   exit if maxCoeffEnergy < 3 × N0 × nRxAnt × nUeLayers
   ```
   Prevents selecting CP intervals that are dominated by noise.

2. **Wiener lambda** (§7.9, Step 5):
   ```
   λ = (antPSD × eigVal) / (antPSD × eigVal + N0)
   ```
   Regularizes the MMSE coefficient to balance signal fidelity vs. noise amplification.

No explicit SNR ratio is ever computed; N0 and `antPSD` (signal power) enter only
as additive terms in the Wiener formula.

### 22.4 LS_ONLY — No Noise Treatment

The LS estimate `H_LS = received / pilot` is written directly without any smoothing
or regularization. There is no noise term and no filter at all.

### 22.5 Comparison Table

| Algorithm | N0 at runtime | SNR used | Where |
|---|---|---|---|
| LEGACY_MMSE | No | 10⁻³ (fixed) | Baked into W offline |
| MULTISTAGE_MMSE | No | 10⁻³ (fixed) | Baked into W offline |
| RKHS | **Yes** | Derived from N0 | Matching pursuit exit + lambda |
| LS_ONLY | No | None | N/A |

---

## 23. Multiple Time Instances (chEstTimeInst)

### 23.1 Purpose

A single 5G NR slot can have up to 4 DMRS time-domain positions
(`CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST = 4`). Processing them as separate time instances
allows downstream blocks (equalizer, SINR estimation) to interpolate the channel
estimate across OFDM symbols, tracking fast time-varying channels.

### 23.2 Time Instance Index

`chEstTimeInst ∈ [0, nTimeChEsts−1]` is stored in `puschRxChEstDynDescr_t`:

```cpp
const uint8_t chEstTimeInst = pDynDescr->chEstTimeInst;
```

`nTimeChEsts` is set to `CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST` in `ch_est_settings.hpp`:

```cpp
nTimeChEsts = CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST;
```

### 23.3 Per-Instance DMRS Symbol Pointer

Each time instance corresponds to a different DMRS symbol (or pair, for `dmrsMaxLen=2`).
The pointer is derived from a flat array `dmrsSymLoc[]` in `cuphyPuschRxUeGrpPrms_t`:

```cpp
const uint8_t* const pDmrsSymPos = &drvdUeGrpPrm.dmrsSymLoc[chEstTimeInst * dmrsMaxLen];
```

- `pDmrsSymPos[0]` — first DMRS symbol position (absolute slot index 0–13)
- `pDmrsSymPos[1]` — second DMRS symbol (only valid when `dmrsMaxLen == 2`)

### 23.4 Output Tensor Indexing

The output tensor `tInfoHEst` has shape `(nRxAnts, nLayers, nSubcarriers, nTimeInst)`.
Each kernel invocation writes to slice `[..., ..., ..., chEstTimeInst]`:

```cpp
tHEst(antIdx, layerIdx, scIdx, chEstTimeInst) = H_est;
```

Similarly, `tInfoDmrsLSEst` (MULTISTAGE intermediate) is indexed by time instance:

```cpp
tInfoDmrsLSEst[dmrsToneIdx/2, layerIdx, bsAntIdx, chEstTimeInst]
```

### 23.5 Delay Accumulation Double-Buffer

The R1 accumulator (§10.2) uses a ping-pong scheme keyed on `dmrsActiveAccumBuf`:

```cpp
// Stage 1 writes to active buffer:
atomicAdd(&pAccum[dmrsActiveAccumBuf].x, sh_R1.x);
// Stage 1 clears inactive buffer (in preparation for next time instance):
pAccum[dmrsActiveAccumBuf ^ 1] = {0.0f, 0.0f};
```

`dmrsActiveAccumBuf` is toggled between time instances to prevent R1 from mixing
contributions of different DMRS symbol positions.

### 23.6 Delay Mean per Time Instance

Each time instance produces its own `delay_mean` value. The output
`tInfoDmrsDelayMean` is a 1D tensor indexed by `[chEstTimeInst]`:

```cpp
tInfoDmrsDelayMean[chEstTimeInst] = sh_delay_mean;
```

This allows downstream blocks to use per-symbol-group delay estimates when computing
time-varying channel interpolation.

---

*Document generated from source analysis of `cuPHY/src/cuphy/ch_est/ch_est.cu`,
`ch_est_types.hpp`, `ch_est_settings.hpp`, and MATLAB reference
`5GModel/nr_matlab/pxsch/pusch_ChEst_LS_delayEst_MMSE.m`,
`derive_chest_mmse_coeff.m`.*
