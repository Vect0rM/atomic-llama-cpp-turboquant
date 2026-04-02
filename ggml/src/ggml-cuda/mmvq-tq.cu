/*
 * Fused mul_mat_vec for TQ4_1S / TQ3_1S weight types.
 *
 * Two-phase approach:
 * Phase 1: WHT rotate activation in 32-element blocks (1 kernel launch)
 * Phase 2: Simple mmvq with pre-rotated activation (centroid × scale only)
 *
 * Amortizes WHT over ALL output rows — one rotation per mul_mat instead
 * of one per block per row.
 *
 * Optimization log (10 versions tested on Qwen2.5-7B TQ4_1S, RTX 5090):
 *   cuBLAS baseline:              20 t/s
 *   V1  per-warp WHT, 4 warps:   60 t/s
 *   V3  shmem activation cache:  33 t/s (syncthreads overhead)
 *   V5  multi-warp per row:      62 t/s
 *   V6  LUT (shmem):             37 t/s (sync overhead)
 *   V7  8 warps clean:           62 t/s
 *   V8  pre-rotation (2-phase):  69 t/s ← BEST
 *   V9  pre-rot + q8_1:          70 t/s (marginal)
 *   V10 4-elem/thread:           57 t/s
 *   V13 8-elem, 4-thread dot:    45 t/s
 *   NR0=2/4/8: all regressed
 */

#include "mmvq-tq.cuh"
#include "turbo-quant.cuh"

#define MMVQ_TQ_NWARPS 8

static __global__ void tq_prerotate_activation(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = val;
}

static __global__ void mul_mat_vec_tq4_1s_fused(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_TQ4_1S + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;

        sum += act * TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

static __device__ __forceinline__ uint8_t tq3_extract_index(const uint8_t * __restrict__ qs, int lane) {
    const int group = lane / 8;
    const int lane_in_group = lane % 8;
    const uint8_t * qp = qs + group * 3;
    const uint32_t packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
    return (packed >> (lane_in_group * 3)) & 7;
}

static __global__ void mul_mat_vec_tq3_1s_fused(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_tq3_1s * x_row = ((const block_tq3_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_TQ3_0 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);

        sum += act * TQ3_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

static float * d_act_rotated = nullptr;
static size_t  d_act_rotated_size = 0;

void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * src0,
                               const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);

    const int ncols_x = src0->ne[0];
    const int nrows_x = src0->ne[1];
    GGML_ASSERT(ncols_x % 32 == 0);

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    const size_t act_bytes = ncols_x * sizeof(float);
    if (act_bytes > d_act_rotated_size) {
        if (d_act_rotated) cudaFree(d_act_rotated);
        cudaMalloc(&d_act_rotated, act_bytes);
        d_act_rotated_size = act_bytes;
    }

    {
        const int n_blocks = ncols_x / 32;
        const int warps_per_block = 4;
        const dim3 block(32, warps_per_block);
        const dim3 grid((n_blocks + warps_per_block - 1) / warps_per_block);
        tq_prerotate_activation<<<grid, block, 0, stream>>>(src1_d, d_act_rotated, ncols_x);
    }

    {
        const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
        const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);

        if (src0->type == GGML_TYPE_TQ4_1S) {
            mul_mat_vec_tq4_1s_fused<<<grid, block, 0, stream>>>(src0_d, d_act_rotated, dst_d, ncols_x, nrows_x);
        } else {
            mul_mat_vec_tq3_1s_fused<<<grid, block, 0, stream>>>(src0_d, d_act_rotated, dst_d, ncols_x, nrows_x);
        }
    }
}
