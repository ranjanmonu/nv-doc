# PUSCH Carrier Frequency Offset (CFO) and Timing Advance (TA) Estimation — Technical Reference

**Source tree:** `cuPHY/src/cuphy/cfo_ta_est/`  
**Correction applied in:** `cuPHY/src/cuphy/channel_eq/channel_eq.cu`  
**MATLAB reference:** `5GModel/nr_matlab/pxsch/detPusch.m`, `apply_equalizer_cfo.m`  
**Spec reference:** 3GPP TS 38.211 (OFDM numerology, DMRS placement)

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Physical Model](#2-physical-model)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Kernel: cfoTaEstLowMimoKernel](#4-kernel-cfotaestlowmimokernel)
5. [CFO Estimation — Detailed Steps](#5-cfo-estimation--detailed-steps)
6. [TA Estimation — Detailed Steps](#6-ta-estimation--detailed-steps)
7. [Weighted Average CFO Filtering](#7-weighted-average-cfo-filtering)
8. [Per-Symbol CFO Phasor Generation](#8-per-symbol-cfo-phasor-generation)
9. [CFO in Hz Conversion](#9-cfo-in-hz-conversion)
10. [CFO Correction in the Equalizer](#10-cfo-correction-in-the-equalizer)
11. [CUDA Parallelism and Memory Model](#11-cuda-parallelism-and-memory-model)
12. [Inter-CTA Synchronization](#12-inter-cta-synchronization)
13. [Key Data Structures and Tensors](#13-key-data-structures-and-tensors)
14. [Template Instantiation Range](#14-template-instantiation-range)
15. [Weighted Average CFO Buffer Pool](#15-weighted-average-cfo-buffer-pool)
16. [MATLAB Reference Equivalence](#16-matlab-reference-equivalence)
17. [End-to-End Data Flow](#17-end-to-end-data-flow)

---

## 1. Purpose and Scope

A UE transmitter with an imperfect local oscillator transmits at frequency `f_c + Δf_CFO`
instead of the target `f_c`. This shifts every OFDM subcarrier by `Δf_CFO`, which appears
as a time-varying phase rotation of the channel estimate across OFDM symbols:

```
H_received[symbol] = H_true × exp(+j × 2π × CFO × symbol_time)
```

Uncorrected CFO:
- Rotates the effective channel between DMRS and data symbols → equalizer coefficient mismatch
- Grows linearly with symbol offset from the DMRS position

This block estimates CFO and Timing Advance (TA) from the multi-time-domain DMRS channel
estimates produced by `ch_est`, then pre-computes a per-symbol correction phasor. The
equalizer applies this phasor when computing equalization coefficients and soft estimates.

**Prerequisites:**
- At least 2 DMRS time instances (`dmrsAddlnPos > 0`, i.e., `AdditionalPosition >= 1`)
- Channel estimation must have already produced `tInfoHEst` for all time instances

---

## 2. Physical Model

### 2.1 CFO — Phase Ramp in Time

Given channel estimates at two consecutive DMRS symbol positions `t₀` and `t₁`:

```
H[t₁] ≈ H[t₀] × exp(+j × 2π × CFO × (t₁ − t₀) × T_sym)
```

The lag-1 time autocorrelation isolates the CFO phase:

```
R_time = Σ_{ant,sc} H*(t₀, ant, sc) × H(t₁, ant, sc)
       ≈ |H|² × exp(+j × 2π × CFO × (t₁ − t₀) × T_sym)
```

where the sum averages out noise. Then:

```
∠R_time = 2π × CFO × (t₁ − t₀) × T_sym
```

Since the DMRS positions are indexed in OFDM symbol units (not seconds), and one OFDM
symbol occupies `T_sym = 1/(Δf × N_FFT)`, the phase per symbol-spacing is:

```
cfoPhase = ∠R_time / (t₁ − t₀)    [radians/symbol]
```

And `CFO = cfoPhase / (2π × T_sym)`.

For data symbol `s` displaced from the first DMRS symbol `t_DMRS₀`:

```
H_effective[s] = H[t_DMRS₀] × exp(+j × cfoPhase × (s − t_DMRS₀))
correction[s]  = exp(−j × cfoPhase × (s − t_DMRS₀))
```

### 2.2 TA — Phase Ramp in Frequency

Timing advance (residual cyclic prefix usage) appears as a linear phase ramp across
subcarriers. The lag-1 frequency autocorrelation estimates this slope:

```
R_freq = Σ_{ant,t,sc} H*(k, ant, t) × H(k+1, ant, t)
       ≈ |H|² × exp(−j × 2π × TA × Δf)

∠R_freq = −2π × TA × Δf
TA = −∠R_freq / (2π × Δf)
```

where `Δf = scsKHz × 1000` Hz is the subcarrier spacing.

---

## 3. Algorithm Overview

```
Inputs:
  tInfoHEst[nRxAnt, nLayers, nSubcarriers, nTimeInst]   channel estimates (all time instances)

Outputs:
  tInfoCfoEst[MAX_ND_SUPPORTED=14, nUes]                CFO phasor per data symbol per UE
  tInfoCfoHz[nUes]                                       CFO in Hz per UE
  tInfoTaEst[nUes]                                       TA in microseconds per UE

Processing steps:
  1. Per-subcarrier phase rotation: H[t+1] × conj(H[t])  (CFO)
                                    H[k+1] × conj(H[k])  (TA)
  2. Warp-level reduction within a thread group tile
  3. Thread-block-level reduction (across tiles for same layer)
  4. Inter-CTA reduction to global accumulator (tCfoPhaseRot, tTaPhaseRot)
  5. Last CTA computes final estimates:
       a. cfoPhase = -atan2(Im, Re) / symbolDistance
       b. Optional weighted average filtering
       c. Per-symbol phasor: cfoEst(s) = exp(j × cfoPhase × (s − t_DMRS₀))
       d. taEst = -taPhase × 1000 / (2π × scsKHz)
```

---

## 4. Kernel: cfoTaEstLowMimoKernel

**Source:** `cfo_ta_est.cu`, function `cfoTaEstLowMimoKernel_v2()` (wrapped by
`cfoTaEstLowMimoKernel`)

### 4.1 Template Parameters

```cpp
template <typename TStorageIn,          // input H storage type  (float or __half)
          typename TStorageOut,         // output phasor type     (float or __half)
          typename TCompute,            // accumulation type      (float)
          uint32_t N_BS_ANTS,           // number of Rx antennas
          uint32_t N_LAYERS,            // number of MIMO layers
          uint32_t N_TIME_CH_EST,       // number of DMRS time instances (2, 3, or 4)
          uint32_t THRD_GRP_TILE_SIZE,  // = 32 (warp size)
          uint32_t N_THRD_GRP_TILES_PER_LAYER,
          uint32_t N_PRB_PER_THRD_BLK> // = 8
```

Constraint: `N_TIME_CH_EST ∈ [2, 4]` — CFO requires at least 2 DMRS positions.

### 4.2 Launch Configuration

```
gridDim  = (ceil(nMaxPrb / N_PRB_PER_THRD_BLK),   nUeGrps)
blockDim = (N_THRDS_PER_LAYER,                      N_LAYERS)
```

where:

| Constant | Value | Derivation |
|---|---|---|
| `N_PRB_PER_THRD_BLK` | 8 | Fixed (each block processes 8 PRBs) |
| `THRD_GRP_TILE_SIZE` | 32 | = warp size |
| `N_THRD_GRP_TILES_PER_LAYER` | `ceil(N_BS_ANTS × 12 / 32)` | Tiles needed to cover all antenna × tone pairs |
| `N_THRDS_PER_LAYER` | `N_THRD_GRP_TILES_PER_LAYER × 32` | Total threads per layer |
| `blockDim.x` | `N_THRDS_PER_LAYER` | Threads covering one layer |
| `blockDim.y` | `N_LAYERS` | One thread group per layer |

Early exit for blocks beyond the UE's PRB allocation:
```cpp
uint32_t nThrdBlksNeeded = div_round_up(nPrb, N_PRB_PER_THRD_BLK);
if (blockIdx.x >= nThrdBlksNeeded) return;
```

### 4.3 Shared Memory

```cpp
__shared__ TComplexCompute shChEstTimePhaseRotSum[N_TIME_CH_EST-1][N_LAYERS];  // CFO accumulator
__shared__ TComplexCompute shChEstFreqPhaseRotSum[N_LAYERS];                   // TA accumulator
__shared__ bool            shIsLastCtaDone;                                    // inter-CTA flag
__shared__ TCompute        shAccumCfoPhase[N_LAYERS];                          // per-layer CFO sum
__shared__ TCompute        shAccumTaPhase[N_LAYERS];                           // per-layer TA sum
__shared__ TComplexCompute smemBlk[N_BS_ANTS × N_LAYERS × 12 × N_TIME_CH_EST]; // H cache
```

`smemBlk` is indexed as `shHEst(bsAntIdx, layerIdx, toneIdx, chEstTimeIdx)`.

### 4.4 Thread Index Decomposition

Each thread is assigned to one antenna, one tone within a PRB, and one layer:

```cpp
const uint32_t bsAntIdx  = thrdIdx % nRxAnt;
const uint32_t toneIdx   = (thrdIdx / nRxAnt) % CUPHY_N_TONES_PER_PRB;  // 0..11
const uint32_t layerIdx2 = (thrdIdx / N_THRDS_PER_LAYER) % nLayers;
```

---

## 5. CFO Estimation — Detailed Steps

### 5.1 Step 1 — Load H into Shared Memory

For each PRB processed by this thread block and each time instance `chEstTimeIdx`:

```cpp
startFreqIdx = (startPrbThisThrdBlk + prbIdx) * 12;   // absolute first subcarrier

shHEst(bsAntIdx, layerIdx2, toneIdx, chEstTimeIdx) =
    type_convert<TComplexCompute>(
        tHEst(bsAntIdx, layerIdx2, startFreqIdx + toneIdx, chEstTimeIdx)
    );
```

### 5.2 Step 2 — Per-Thread Phase Rotation (Time Domain — CFO)

For each consecutive pair of DMRS time instances:

```cpp
// For each chEstTimeIdx in [0, nCHEST-2]:
TComplexCompute chEst0 = shHEst(bsAntIdx, layerIdx2, toneIdx, chEstTimeIdx);
TComplexCompute chEst1 = shHEst(bsAntIdx, layerIdx2, toneIdx, chEstTimeIdx + 1);

chEstTimePhaseRot[chEstTimeIdx] = chEst1 * cuConj(chEst0);
// = H[t+1] × H*(t) = |H|² × exp(+j × 2π × CFO × ΔT)
```

Each thread computes this for its assigned (antenna, tone, layer) combination.

### 5.3 Step 3 — Warp-Level Reduction (Within Tile)

Using cooperative group tile of size `THRD_GRP_TILE_SIZE = 32`:

```cpp
chEstTimePhaseRotSum[chEstTimeIdx] =
    thrdGrpAllReduceSum<TComplexCompute, THRD_GRP_TILE_SIZE>(
        layerThrdTile, chEstTimePhaseRot[chEstTimeIdx]
    );
```

`thrdGrpAllReduceSum` performs a butterfly reduction using `shfl_xor`:

```cpp
for (int i = tileSize/2; i > 0; i /= 2) {
    sum.x += thisThrdGrp.shfl_xor(cuReal(sum), i);
    sum.y += thisThrdGrp.shfl_xor(cuImag(sum), i);
}
```

After this, every thread in the tile holds the tile-wide sum.

### 5.4 Step 4 — Thread-Block-Level Reduction

Lane 0 of each thread group tile atomically adds to the shared-memory accumulator:

```cpp
if (layerThrdGrpThrdIdx < nCHEST-1) {
    uint32_t chEstTimeIdx2 = layerThrdGrpThrdIdx;
    atomicAdd(&shChEstTimePhaseRotSum[chEstTimeIdx2][layerIdx2].x,
              cuReal(chEstTimePhaseRotSum[chEstTimeIdx2]));
    atomicAdd(&shChEstTimePhaseRotSum[chEstTimeIdx2][layerIdx2].y,
              cuImag(chEstTimePhaseRotSum[chEstTimeIdx2]));
}
```

After `__syncthreads()`, `shChEstTimePhaseRotSum[t][layer]` holds the sum of
`H[t+1] × H*(t)` across all antennas and all tones processed by this block.

### 5.5 Step 5 — Inter-CTA Reduction to Global Memory

Each thread block contributes its per-layer sums to the global accumulator
`tCfoPhaseRot[chEstTimeIdx, layerIdx, ueGrpIdx]`:

```cpp
if (thrdIdx < nLayers*(nCHEST-1)) {
    volatile TComplexCompute& timePhaseRotSum =
        tCfoPhaseRot(chEstTimeIdx1, layerIdx1, ueGrpIdx);
    atomicAdd(const_cast<float*>(&timePhaseRotSum.x),
              cuReal(shChEstTimePhaseRotSum[chEstTimeIdx1][layerIdx1]));
    atomicAdd(const_cast<float*>(&timePhaseRotSum.y),
              cuImag(shChEstTimePhaseRotSum[chEstTimeIdx1][layerIdx1]));
}
```

`__threadfence()` is called before the counter increment to ensure global visibility.

### 5.6 Step 6 — Last-CTA Detection

```cpp
if (0 == thrdIdx) {
    __threadfence();
    uint32_t syncCnt = atomicInc(&interCtaSyncCnt, nThrdBlksNeeded);
    shIsLastCtaDone = (syncCnt == (nThrdBlksNeeded - 1));
}
__syncthreads();
```

Only the last CTA to complete for this UE group proceeds to compute the final estimate.
All other CTAs exit.

### 5.7 Step 7 — Final CFO Phase Extraction (Last CTA Only)

For each time instance pair `(t, t+1)` and each layer:

```cpp
volatile TComplexCompute& timePhaseRotSum =
    tCfoPhaseRot(chEstTimeIdx1, layerIdx1, ueGrpIdx);

uint32_t firstDmrsSymbPosIdx  = chEstTimeIdx1 * dmrsMaxLen;
uint32_t secondDmrsSymbPosIdx = (chEstTimeIdx1 + 1) * dmrsMaxLen;

TCompute cfoPhase =
    -atan2f(cuImag(timePhaseRotSum), cuReal(timePhaseRotSum))
    / (float)(pPilotSymbPos[secondDmrsSymbPosIdx] - pPilotSymbPos[firstDmrsSymbPosIdx]);
```

`pPilotSymbPos[secondDmrsSymbPosIdx] - pPilotSymbPos[firstDmrsSymbPosIdx]` is the
DMRS symbol distance in OFDM symbol units (typically 4, 7, or 11 symbols for
`AdditionalPosition = 1, 2, 3`).

The `cfoPhase` (radians per OFDM symbol) is accumulated per UE (across layers):

```cpp
uint8_t ueIdx = pUeGrpLayerToUeIdx[layerIdx1];
atomicAdd(&shAccumCfoPhase[ueIdx], cfoPhase);
```

After `__syncthreads()`, `shAccumCfoPhase[ueIdx]` holds the sum of `cfoPhase`
contributions from all `nLayers[ueIdx] × (nCHEST-1)` layer–time-pair combinations.

---

## 6. TA Estimation — Detailed Steps

### 6.1 Step 1 — Per-Thread Frequency Phase Rotation

For each subcarrier pair within the PRB (skipping last tone):

```cpp
// For each chEstTimeIdx and each bsAntIdx:
if ((layerThrdIdx < (CUPHY_N_TONES_PER_PRB * nRxAnt)) &&
    (toneIdx < (CUPHY_N_TONES_PER_PRB - 1)))
{
    TComplexCompute chEst0 = shHEst(bsAntIdx, layerIdx2, toneIdx,     chEstTimeIdx);
    TComplexCompute chEst1 = shHEst(bsAntIdx, layerIdx2, toneIdx + 1, chEstTimeIdx);
    chEstFreqPhaseRotSum += (chEst1 * cuConj(chEst0));
    // = H[k+1] × H*(k) = |H|² × exp(-j × 2π × TA × Δf)
}
```

This accumulates across all time instances and antennas.

### 6.2 Step 2 — Warp and Block Reduction

Same two-stage reduce as CFO (§5.3–5.4), but writes to `shChEstFreqPhaseRotSum[layerIdx]`
and then to global `tTaPhaseRot[layerIdx, ueGrpIdx]`.

### 6.3 Step 3 — TA Phase Extraction (Last CTA)

```cpp
if (thrdIdx < nLayers) {
    volatile TComplexCompute& freqPhaseRotSum = tTaPhaseRot(layerIdx0, ueGrpIdx);
    TCompute taPhase = atan2f(cuImag(freqPhaseRotSum), cuReal(freqPhaseRotSum));
    uint8_t ueIdx = pUeGrpLayerToUeIdx[layerIdx0];
    atomicAdd(&shAccumTaPhase[ueIdx], taPhase);
}
```

Note the sign: `+atan2f` for TA vs. `-atan2f` for CFO — TA causes a positive
frequency-domain phase slope, CFO a positive time-domain phase slope.

### 6.4 Step 4 — TA Estimate in Microseconds

```cpp
if (thrdIdx < nUes) {
    TCompute avgTaPhase = shAccumTaPhase[ueIdx] / pNUeLayers[ueIdx];
    TCompute taEst = -avgTaPhase * 1000 / (TWO_PI * deltaFKHz);
    tTaEst(pAbsUeIdxs[ueIdx]) = type_convert<TStorageOut>(taEst);
}
```

`deltaFKHz = scsKHz` (e.g., 15, 30, 60, 120). The factor `1000/scsKHz` converts from
radians/(subcarrier spacing) to microseconds:

```
TA [μs] = -avgTaPhase × 1000 / (2π × scsKHz)
        = -avgTaPhase / (2π × scsKHz × 1000)  × 10⁶
```

---

## 7. Weighted Average CFO Filtering

When `enableWeightedAverageCfo = 1`, each UE maintains a **persistent complex accumulator**
across slots. This acts as an exponential moving average of the CFO estimate, weighted by
both a forget coefficient `α` and the current pre-equalizer SINR.

### 7.1 Buffer Structure

Each UE has a 2-float buffer `[buffer[0], buffer[1]]` representing a complex number:
`buffer[0] + j×buffer[1]`. This buffer persists across slots via the `WAvgCfoPool`.

### 7.2 Update Rule

```cpp
float foForgetCoeff = drvdUeGrpPrms.foForgetCoeff[ueIdx];   // α, typically ~0.95
float cfoAnglecurr  = -shAccumCfoPhase[ueIdx]
                     / (pNUeLayers[ueIdx] * (nCHEST-1));     // instantaneous estimate

// SNR-weighted current contribution:
float coeffCurr = (1.0f - foForgetCoeff) * nPrb
                  * powf(10.0f, (float)tSinrPreEq(ueIdx) / 10.0f);

// Exponential moving average in Cartesian coordinates:
buffer[0] = foForgetCoeff * buffer[0] + coeffCurr * cosf(cfoAnglecurr);
buffer[1] = foForgetCoeff * buffer[1] + coeffCurr * sinf(cfoAnglecurr);

// Recover filtered angle:
float cfoAngleFiltered = -atan2f(buffer[1], buffer[0]);

// Inject back as accumulated phase for downstream phasor generation:
shAccumCfoPhase[ueIdx] = cfoAngleFiltered * (pNUeLayers[ueIdx] * (nCHEST-1));
```

### 7.3 Why Cartesian Averaging?

Averaging phases directly (`α × θ_old + (1−α) × θ_new`) wraps at ±π and is sensitive
to sign. Averaging the complex phasor `exp(j×θ)` in Cartesian form is numerically stable
and correctly handles wrap-around.

### 7.4 SNR Weighting

The current-slot contribution is scaled by:
```
coeffCurr = (1−α) × nPrb × SNR_linear
```

Higher SNR → higher weight for the current measurement. This reduces reliance on noisy
estimates at low SNR and trusts the history more.

**MATLAB equivalent** (`detPusch.m`):
```matlab
foCompensationBuffer(idxUe) =
    foForgetCoeff * foCompensationBuffer(idxUe)
    + (1 - foForgetCoeff) * nPrb * 10^(sinrdB(idxUe)/10) * exp(j*2*pi*cfo_est_current);
cfo_est_filtered = (1/2/pi) * angle(foCompensationBuffer(idxUe));
```

---

## 8. Per-Symbol CFO Phasor Generation

After the final CFO phase (possibly filtered) is in `shAccumCfoPhase[ueIdx]`, the kernel
generates a correction phasor for every OFDM symbol in the slot (`MAX_ND_SUPPORTED = 14`):

```cpp
for (int iterIdx = 0; iterIdx < nCfoEstIter; ++iterIdx) {
    uint32_t idx    = (iterIdx * nThrds) + thrdIdx;
    int32_t  symbIdx = idx % MAX_ND_SUPPORTED;
    uint8_t  ueIdx   = idx / MAX_ND_SUPPORTED;

    // Average over layers and time estimates; then scale by symbol offset:
    TCompute cfoAngle =
        shAccumCfoPhase[ueIdx]
        * (symbIdx - pPilotSymbPos[0])          // symbol offset from first DMRS
        / (pNUeLayers[ueIdx] * (nCHEST - 1));  // normalize by #layers × #pairs

    TComplexCompute cfoEst = { cosf(cfoAngle), sinf(cfoAngle) };
    tCfoEst(symbIdx, ueIdx) = type_convert<TComplexStorageOut>(cfoEst);
}
```

**Interpretation:**
- `shAccumCfoPhase[ueIdx]` is the total accumulated phase from `nLayers × (nCHEST−1)`
  contributions. Dividing by this normalizes to the per-symbol phase per unit offset.
- Multiplying by `(symbIdx − pPilotSymbPos[0])` gives the CFO phase at symbol `symbIdx`.
- The phasor `exp(j × cfoAngle)` is the forward rotation at symbol `s`.
- The correction applied by the equalizer is the **conjugate**: `exp(−j × cfoAngle)`.

The tensor `tCfoEst` has shape `(MAX_ND_SUPPORTED=14, MAX_N_UE_PER_UE_GRP)`, pre-populated
for every possible data symbol index.

---

## 9. CFO in Hz Conversion

```cpp
if ((symbIdx - pPilotSymbPos[0]) == 1) {
    tCfoHz(pAbsUeIdxs[ueIdx]) =
        -cfoAngle * drvdUeGrpPrms.scsKHz * 1e6f
        / (2.0f * M_PI * 15.0f * 71.35f);
}
```

The factor `15 × 71.35e-6 = 1.07025e-3` converts from per-symbol phase to Hz:

```
CFO [Hz] = −cfoPhase [rad/symbol] × (scsKHz × 1000) [Hz/cycle]
           ÷ (2π [rad/cycle]) ÷ T_sym [s/symbol]

T_sym = 1 / (15e3 × 2^μ) + CP_overhead ≈ 71.35 μs  (normal CP, μ=0)

CFO [Hz] = −cfoAngle × scsKHz × 1e6 / (2π × 15 × 71.35)
```

The formula uses the symbol time for numerology μ=0 (`15 × 71.35e-6 s`), then scales
by `scsKHz/15` for other numerologies via the `scsKHz` factor.

---

## 10. CFO Correction in the Equalizer

**Source:** `channel_eq.cu`

The correction is applied during the loading of channel estimates `tHEst` into the
equalizer's shared memory. Two application points exist:

### 10.1 Equalization Coefficient Computation

```cpp
bool enableCfoCorrection = (0 != drvdUeGrpPrms.enableCfoCorrection);
const bool doCfo   = enableCfoCorrection && (chEqTimeInstIdx > 0);
const int  dmrsIdx = dmrsSymLoc[chEqTimeInstIdx * dmrsMaxLen];  // DMRS symbol for this time instance

TComplexCompute Hval = type_convert<TComplexCompute>(tH(r, c, ABS_FREQ_IDX, chEqTimeInstIdx));
if (doCfo) {
    Hval = cuCmul(Hval,
                  type_convert<TComplexCompute>(tCfoEst(dmrsIdx, pUeGrpLayerToUeIdx[c])));
}
shH(r, c) = Hval;
```

**Key points:**
- `chEqTimeInstIdx = 0` (first DMRS position) is **never corrected** — it is the
  reference point against which the CFO was estimated.
- For `chEqTimeInstIdx > 0`, the channel estimate is multiplied by
  `tCfoEst(dmrsSymPos, ueIdx)`, which rotates `H` back to the reference phase.
- This must be done before computing the MMSE equalizer coefficients `W = (H^H × H + noise)^{-1} × H^H`.

**MATLAB equivalent** (`detPusch.m`):
```matlab
% correct channel estimate by CFO
for posDmrs = 2:AdditionalPosition+1
    for idxUe = 1:nUe
        layerIdx = find(layer2Ue == (idxUe-1));
        ue_cfo = cfo_est(end, layerIdx);   % final or per-instance depending on mode
        H_est{posDmrs}(:, layerIdx, :) =
            H_est{posDmrs}(:, layerIdx, :)
            .* exp(-j*2*pi * ue_cfo * (dmrsIdx{posDmrs}(1) - dmrsIdx{1}(1)));
    end
end
```

### 10.2 Soft Demapper Application

```cpp
// In the soft demapper kernel, for each data symbol:
if (enableCfoCorrection) {
    softEst = cuCmul(softEst,
                     type_convert<TComplexCompute>(tCfoEst(symIdx, pUeGrpLayerToUeIdx[layerIdx])));
}
```

The equalized data estimate `X̂ = W × Y` is multiplied by `tCfoEst(s, ue)` to compensate
for the frequency rotation at each data symbol `s`.

**MATLAB equivalent** (`apply_equalizer_cfo.m`):
```matlab
for f = 1:Nf
    for t = 1:nSym_data
        X_est(:,f,t) = W(:,:,f) * Y(:,f,symIdx_data(t));
        for ll = 1:L_UE
            X_est(ll,f,t) *= exp(-j*2*pi * cfo(ll) * (symIdx_data(t) - symIdx_dmrs(1)));
        end
    end
end
```

---

## 11. CUDA Parallelism and Memory Model

### 11.1 Thread Mapping

```
Thread index decomposition within blockDim = (N_THRDS_PER_LAYER, N_LAYERS):
  threadIdx.y = layerIdx          (one row of threads per layer)
  threadIdx.x = bsAntIdx + (toneIdx × nRxAnt) mod N_THRDS_PER_LAYER
```

For `N_BS_ANTS = 4`, `N_LAYERS = 2`, `CUPHY_N_TONES_PER_PRB = 12`:
- `N_THRD_GRP_TILES_PER_LAYER = ceil(4 × 12 / 32) = 2`
- `N_THRDS_PER_LAYER = 2 × 32 = 64`
- `blockDim = (64, 2, 1)` — 128 threads per block

### 11.2 PRB Processing Loop

Each thread block iterates over up to `N_PRB_PER_THRD_BLK = 8` PRBs:

```cpp
for (int32_t prbIdx = 0; prbIdx < nPrbThisThrdBlk; ++prbIdx) {
    // Load this PRB's H into shHEst (all time instances)
    // Compute phase rotations
    // Warp-level reduce → shChEstTimePhaseRotSum, shChEstFreqPhaseRotSum
}
```

### 11.3 Memory Access Coalescing

- `tHEst` is shaped `[nRxAnt, nLayers, nSubcarriers, nTimeInst]` (column-major innermost)
- Thread loading pattern: threads differ by `bsAntIdx` for consecutive `threadIdx.x`
  values, which are the innermost stride → coalesced access

### 11.4 Two-Level Reduction Summary

| Level | Structure | How |
|---|---|---|
| Warp | `layerThrdTile` (32 threads) | `shfl_xor` butterfly |
| Thread block | `shChEstTimePhaseRotSum[]` | `atomicAdd` to SMEM |
| Cross-block | `tCfoPhaseRot` (global) | `atomicAdd` to GMEM |
| Final | Last CTA reads global sum | `-atan2f` |

---

## 12. Inter-CTA Synchronization

CUDA does not provide a native cross-block barrier within a single kernel launch.
The implementation uses a software counter in global memory:

```cpp
// Each block's thread 0 increments the counter after fencing:
__threadfence();   // ensure all atomicAdds above are globally visible
uint32_t syncCnt = atomicInc(&interCtaSyncCnt, nThrdBlksNeeded);

// Last CTA check:
shIsLastCtaDone = (syncCnt == (nThrdBlksNeeded - 1));
__syncthreads();   // broadcast to all threads in this block

if (shIsLastCtaDone) {
    // Only this block computes the final estimate
}
```

**Correctness guarantee:**
1. `__threadfence()` ensures the `atomicAdd` to `tCfoPhaseRot` is visible globally
   before the counter increment.
2. `atomicInc` with limit `nThrdBlksNeeded` auto-wraps back to 0, so the counter is
   ready for the next kernel invocation without explicit reset.
3. `shIsLastCtaDone` is written by thread 0 and read by all threads after `__syncthreads()`.

**Important:** `tInterCtaSyncCnt` must be pre-initialized to 0 before each kernel launch.
The kernel description in the comment header explicitly states this requirement.

---

## 13. Key Data Structures and Tensors

### 13.1 `cuphyPuschRxUeGrpPrms_t` — CFO/TA-Relevant Fields

| Field | Type | Description |
|---|---|---|
| `enableCfoCorrection` | `uint8_t` | Enable CFO correction in equalizer |
| `enableWeightedAverageCfo` | `uint8_t` | Enable IIR filtered CFO |
| `enableToEstimation` | `uint8_t` | Enable TA (timing offset) estimation |
| `foForgetCoeff[]` | `float[MAX_N_UE]` | Per-UE IIR forget coefficient α |
| `nUes` | `uint8_t` | Number of UEs in this group |
| `nUeLayers[]` | `uint8_t[MAX_N_UE]` | Layers per UE |
| `ueGrpLayerToUeIdx[]` | `uint8_t[]` | Layer → UE index mapping |
| `ueIdxs[]` | `uint16_t[]` | Absolute UE indices |
| `scsKHz` | `uint32_t` | Subcarrier spacing (15/30/60/120 kHz) |
| `dmrsAddlnPos` | `uint8_t` | Additional DMRS positions (0=1 symbol, 1=2, ...) |
| `dmrsSymLoc[]` | `uint8_t[]` | DMRS symbol positions within slot |
| `dmrsMaxLen` | `uint8_t` | Number of DMRS symbols per time instance (1 or 2) |
| `tInfoHEst` | `cuphyTensorInfo4_t` | Input channel estimates `[nAnt, nLayers, nSc, nTime]` |
| `tInfoCfoEst` | `cuphyTensorInfo2_t` | Output CFO phasor `[MAX_ND_SUPPORTED=14, nUes]` |
| `tInfoCfoHz` | `cuphyTensorInfo1_t` | Output CFO in Hz `[nUes]` |
| `tInfoTaEst` | `cuphyTensorInfo1_t` | Output TA in μs `[nUes]` |
| `tInfoCfoPhaseRot` | `cuphyTensorInfo3_t` | Inter-CTA CFO accumulator `[nTimeInst-1, nLayers, nUeGrps]` |
| `tInfoTaPhaseRot` | `cuphyTensorInfo3_t` | Inter-CTA TA accumulator `[nLayers, nUeGrps]` |
| `tInfoCfoTaEstInterCtaSyncCnt` | `cuphyTensorInfo1_t` | CTA counter (must be pre-zeroed) |
| `tInfoSinrPreEq` | `cuphyTensorInfo1_t` | Pre-equalizer SINR in dB (for weighted avg) |

### 13.2 Constants

| Constant | Value | Description |
|---|---|---|
| `CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS` | 1 | Max CFO estimation het-cfgs |
| `CUPHY_PUSCH_RX_CFO_CHECK_THRESHOLD` | 2.0 | CFO validity check threshold |
| `MAX_ND_SUPPORTED` | 14 | Maximum OFDM symbols per slot |
| `N_PRB_PER_THRD_BLK` | 8 | PRBs processed per CUDA thread block |

---

## 14. Template Instantiation Range

The kernel is pre-compiled for all combinations of:

| Parameter | Values |
|---|---|
| `N_BS_ANTS` | 1, 2, 4, 8 |
| `N_LAYERS` | 1, 2, 3, 4, 5, 6, 7, 8 |
| `N_TIME_CH_EST` | 2, 3, 4 |
| `TStorageIn` | `float`, `__half` |
| `TStorageOut` | `float`, `__half` |

Not all combinations are instantiated — kernel selection is in `kernelSelectL0` which
enumerates only `(nBSAnts ≤ 8) × (nLayers ≤ 8)` for the "Low MIMO" regime.

Selection flow:
```
kernelSelectL2(nBSAnts, nLayers, nDmrsAddlnPos, hEstType, cfoEstType)
    → kernelSelectL1<TStorageIn, TStorageOut, TCompute>(...)
    → kernelSelectL0<..., N_TIME_CH_EST = nDmrsAddlnPos+1>(...)
    → cfoTaEstLowMimo<..., N_BS_ANTS, N_LAYERS, N_TIME_CH_EST>(...)
```

---

## 15. Weighted Average CFO Buffer Pool

**Source:** `cuPHY-CP/cuphydriver/include/wavgcfo_pool.hpp`

### 15.1 Buffer Layout

Each UE has a 2-float (8-byte) buffer:
```
buffer[0] = Re(Σ w_i × exp(j×θ_i))   // in-phase accumulator
buffer[1] = Im(Σ w_i × exp(j×θ_i))   // quadrature accumulator
```

### 15.2 Pool Structures

| Class | Description |
|---|---|
| `WAvgCfoBuffer` | One 2-float buffer per UE |
| `WAvgCfoCache` | Maps `(RNTI, cell_id)` → buffer (for fast reuse) |
| `WAvgCfoPool` | Fixed-size pool of 1024 buffers (2 KB total) |
| `WAvgCfoPoolManager` | Combines pool + cache; handles allocation/deallocation |

### 15.3 Buffer Lifecycle

1. On UE scheduling: allocate from pool, cache by `(RNTI, cell_id)`
2. Pass buffer pointer `pFoCompensationBuffers[absUeIdx]` to kernel
3. Kernel reads old value, applies IIR update, writes back
4. Buffer persists across slots until UE departs
5. Cleanup: when pool occupancy > 70%, evict unused entries

### 15.4 Why Persistent Buffers?

The weighted average filter has memory — it depends on the history of CFO estimates.
Without persistence, each slot would start fresh and the averaging would not converge.
The pool provides a simple key-value store keyed on UE identity.

---

## 16. MATLAB Reference Equivalence

### 16.1 CFO Estimation (`detPusch.m`, lines 892–939)

```matlab
accum = zeros(AdditionalPosition, nLayer);
for posDmrs = 1:AdditionalPosition
    for ii = 1:nAnt
        for jj = scIdxs
            for ll = 1:nLayer
                accum(posDmrs, ll) +=
                    H_est{posDmrs+1}(ii, ll, jj) * conj(H_est{posDmrs}(ii, ll, jj));
            end
        end
    end
end

for posDmrs = 1:AdditionalPosition
    dist_dmrs = dmrsIdx{posDmrs+1}(1) - dmrsIdx{posDmrs}(1);
    cfo_est(posDmrs, :) = (1/2/pi) * angle(accum(posDmrs, :)) / dist_dmrs;
end
```

MATLAB uses `angle()` = `atan2(imag, real)` without the negative sign; the sign appears
later when applying the correction. CUDA uses `-atan2f()` directly.

### 16.2 CFO to Hz (`detPusch.m`, line 923)

```matlab
cfo_est_Hz = cfo_est * (delta_f/15e3) / (71.35e-6);
```

This is equivalent to:
```
cfoHz = cfoAngle [rad/symbol] × (Δf/15kHz) / T_sym_15kHz
      = cfoAngle × scsKHz/15 / (71.35e-6)
      = cfoAngle × scsKHz × 1e6 / (15 × 71.35)
```

Consistent with the CUDA formula (with opposite sign convention).

### 16.3 Weighted Average (`detPusch.m`, lines 932–938)

```matlab
if enableWeightedAverageCfo == 1
    for idxUe = 1:nUe
        layerIdx = find(layer2Ue == (idxUe-1));
        foCompensationBuffer(idxUe) =
            foForgetCoeff(idxUe) * foCompensationBuffer(idxUe)
            + (1 - foForgetCoeff(idxUe)) * nPrb * 10^(sinrdB(idxUe)/10)
            * exp(j * 2*pi * cfo_est(1, layerIdx(1)));
        cfo_est(1:AdditionalPosition, layerIdx) =
            (1/2/pi) * angle(foCompensationBuffer(idxUe));
    end
end
```

### 16.4 CFO Correction on Channel Estimate (`detPusch.m`, lines 977–986)

```matlab
for posDmrs = 2:AdditionalPosition+1
    for idxUe = 1:nUe
        layerIdx = find(layer2Ue == (idxUe-1));
        ue_cfo = cfo_est(end, layerIdx);  % use last (most current) estimate
        H_est{posDmrs}(:, layerIdx, :) =
            H_est{posDmrs}(:, layerIdx, :)
            .* exp(-j * 2*pi * ue_cfo * (dmrsIdx{posDmrs}(1) - dmrsIdx{1}(1)));
    end
end
```

### 16.5 CFO Correction on Data (`apply_equalizer_cfo.m`)

```matlab
for f = 1:Nf
    for t = 1:nSym_data
        X_est(:,f,t) = W(:,:,f) * Y(:,f,symIdx_data(t));
        for ll = 1:L_UE
            X_est(ll,f,t) *=
                exp(-j * 2*pi * cfo(ll) * (symIdx_data(t) - symIdx_dmrs(1)));
        end
    end
end
```

---

## 17. End-to-End Data Flow

```
                ┌────────────────────────────────────────────────────┐
                │  tHEst [nRxAnt, nLayers, nSubcarriers, nTimeInst]  │
                │  (produced by ch_est for all DMRS time instances)   │
                └──────────────────┬─────────────────────────────────┘
                                   │
                ┌──────────────────▼─────────────────────────────────┐
                │  cfoTaEstLowMimoKernel                              │
                │                                                      │
                │  Per-thread:                                         │
                │    CFO: chEstTimePhaseRot[t] = H[t+1] × H*(t)       │
                │    TA:  chEstFreqPhaseRot   += H[k+1] × H*(k)       │
                │                                                      │
                │  Warp-tile reduce (shfl_xor, 32 threads)            │
                │                                                      │
                │  Thread-block reduce (atomicAdd to SMEM)             │
                │    → shChEstTimePhaseRotSum[t][layer]                │
                │    → shChEstFreqPhaseRotSum[layer]                   │
                │                                                      │
                │  Inter-CTA reduce (atomicAdd to GMEM)               │
                │    → tCfoPhaseRot[t, layer, ueGrp]                  │
                │    → tTaPhaseRot[layer, ueGrp]                      │
                │                                                      │
                │  Last CTA only:                                      │
                │    cfoPhase = -atan2(Im, Re) / symbolDistance        │
                │    taPhase  = +atan2(Im, Re)                         │
                │    Average over layers → shAccumCfoPhase[ueIdx]      │
                │                                                      │
                │    [If enableWeightedAverageCfo]:                    │
                │    buffer = α×buffer + (1−α)×nPrb×SNR×exp(j×θ)     │
                │    cfoPhase = angle(buffer)                          │
                │                                                      │
                │    Per-symbol phasor:                                │
                │    cfoEst(s, ue) = exp(j × cfoPhase × (s − t₀))    │
                │                                                      │
                │    TA in μs:                                         │
                │    taEst = -avgTaPhase×1000/(2π×scsKHz)             │
                └──────────┬───────────────────────┬──────────────────┘
                           │                        │
              ┌────────────▼────────┐   ┌──────────▼──────────┐
              │  tCfoEst             │   │  tCfoHz / tTaEst    │
              │  [14, nUes]          │   │  scalar per UE      │
              │  phasor per symbol   │   │  reported to L2/L3  │
              └────────────┬────────┘   └─────────────────────┘
                           │
                ┌──────────▼──────────────────────────────────────────┐
                │  channel_eq kernel                                   │
                │                                                      │
                │  Coefficient computation (for chEqTimeInstIdx > 0): │
                │    H_corrected = H × tCfoEst(dmrsSymPos, ueIdx)     │
                │    (rotates channel estimate to reference phase)     │
                │                                                      │
                │  Soft demapper (for each data symbol s):            │
                │    X_est ×= tCfoEst(s, ueIdx)                       │
                │    (compensates for CFO rotation at data symbol)     │
                └─────────────────────────────────────────────────────┘
```

---

*Document generated from source analysis of `cuPHY/src/cuphy/cfo_ta_est/cfo_ta_est.cu`,
`cfo_ta_est.hpp`, `channel_eq/channel_eq.cu`, MATLAB reference
`5GModel/nr_matlab/pxsch/detPusch.m`, `apply_equalizer_cfo.m`,
and `cuPHY-CP/cuphydriver/include/wavgcfo_pool.hpp`.*
