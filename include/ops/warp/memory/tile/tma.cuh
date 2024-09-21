#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
namespace tma {

/* ----------   Create tensor map descriptor (HOST)  ---------- */

/**
* @brief Creates a tensor map for the given source tensor.
*
* This function creates a tensor map (CUtensorMap) for the specified source shared tile type. The tensor map
* is used to describe the shape and layout of the tensor in memory. The function sets up the tensor
* map based on the provided source tensor pointer and the layout specified by the ST template parameter.
*
* @tparam ST The source tensor type, which must be TMA-compatible.
* @tparam blocks_height The number of tiles present on the height axis in global memory.
* @tparam blocks_width The number of tiles present on the width axis in global memory. Defaults to 1.
* @param tma_map Pointer to the CUtensorMap object to be initialized.
* @param src Pointer to the source tensor data in global memory.
*/
template<ducks::st::all ST>
__host__ static inline void create_tensor_map(CUtensorMap *tma_map, const typename ST::dtype *src, int blocks_height, int blocks_width=1) {
    using dtype = typename ST::dtype;
    
    constexpr uint32_t  tma_dim = 3;
    void *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        std::is_same_v<dtype, half>  ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 :
        std::is_same_v<dtype, float> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
        CU_TENSOR_MAP_SWIZZLE_NONE
    );

    uint64_t gmem_shape [4] = {0, 0, 0, 0};
    uint64_t gmem_stride[3] = {0, 0, 0};
    uint32_t smem_shape [4] = {0, 0, 0, 0};
    uint32_t smem_stride[4] = {1, 1, 1, 1};

              uint64_t global_tile_height = blocks_height * ST::rows;
              uint64_t global_tile_width  = blocks_width * ST::cols; 
    constexpr uint64_t shared_tile_height = ST::rows; 
    constexpr uint64_t shared_tile_width  = ST::cols;

    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);

    gmem_shape[0] = swizzle_elements;
    gmem_shape[1] = global_tile_height;
    gmem_shape[2] = global_tile_width / swizzle_elements;

    gmem_stride[0] = global_tile_width * sizeof(dtype);
    gmem_stride[1] = ST::swizzle_bytes;

    smem_shape[0] = swizzle_elements;
    smem_shape[1] = shared_tile_height;
    smem_shape[2] = shared_tile_width / swizzle_elements;

    // ensure that the global address is always 16-byte aligned 
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);

    assert(gmem_stride[0] % 16 == 0); // gmem_stride[0] elements must be a multiple of 16B
    assert(gmem_stride[1] % 16 == 0); // gmem_stride[1] elements must be a multiple of 16B
    assert(gmem_stride[2] % 16 == 0); // gmem_stride[2] elements must be a multiple of 16B

    assert(smem_shape[0] <= 256); // smem_shape[0] elements must be <= 256
    assert(smem_shape[1] <= 256); // smem_shape[1] elements must be <= 256
    assert(smem_shape[2] <= 256); // smem_shape[2] elements must be <= 256
    assert(smem_shape[3] <= 256); // smem_shape[3] elements must be <= 256

    assert((smem_shape[0]*sizeof(dtype)) % 16 == 0); // if wgmma_interleave is none, then smem_shape[0] * sizeof(dtype) must be a multiple of 16B

    assert(smem_stride[0] <= 8); // smem_stride[0] must be less <= 8
    assert(smem_stride[1] <= 8); // smem_stride[1] must be less <= 8
    assert(smem_stride[2] <= 8); // smem_stride[2] must be less <= 8
    assert(smem_stride[3] <= 8); // smem_stride[3] must be less <= 8

    assert(smem_stride[0] == 1); // smem_stride[0] is ignored when wgmma_interleave is none

    if constexpr (tma_interleave == CU_TENSOR_MAP_INTERLEAVE_NONE && tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        assert(smem_shape[0] * sizeof(dtype) <= ST::swizzle_bytes);
    }

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    CUresult result = cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        tma_swizzle,
        tma_l2Promotion,
        tma_oobFill);


    const char *error_string;
    CUresult res = cuGetErrorString(result, &error_string);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Error: " << error_string << std::endl;
    }
}

/**
* @brief Allocates on the GPU and initializes a tensor map for the given source tensor.
*
* This function creates a tensor map (CUtensorMap) for the specified source shared tile type. The tensor map
* is used to describe the shape and layout of the tensor in memory. The function sets up the tensor
* map based on the provided source tensor pointer and the layout specified by the ST template parameter.
*
* @tparam ST The source tensor type, which must be TMA-compatible.
* @tparam blocks_height The number of tiles present on the height axis in global memory.
* @tparam blocks_width The number of tiles present on the width axis in global memory. Defaults to 1.
* @param src Pointer to the source tensor data in global memory.
* @returns Pointer to the CUtensorMap object to be initialized.
*/
template<ducks::st::all ST>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(const typename ST::dtype *src, int blocks_height, int blocks_width=1) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host; // put it on the stack, why not.
    create_tensor_map<ST>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

/* ----------   Prefetch Tensor Map  ---------- */

/**
 * @brief Prefetches data from global memory into a shared memory tile, along with the tensormap.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] tile_row_idx The row index of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the requested tile. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void prefetch(ST &dst, void const* const src_tma_map, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid()) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (dst.rows);
        int32_t crd2 = tile_col_idx * (dst.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
            " [%0, {%1, %2, %3}];"
            :
            : "l"(tma_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/* ----------   Async load and store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously stores data into global memory from a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/* ----------   Async reduction + store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously performs an add reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous add reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_add_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/**
 * @brief Asynchronously performs an min reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous min reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_min_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/**
 * @brief Asynchronously performs an max reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous max reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row index of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void store_max_async(void *dst_tma_map, const ST &src, int tile_row_idx, int tile_col_idx=0) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (src.rows);
        int32_t crd2 = tile_col_idx * (src.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The barrier used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row index of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the requested tile. This is in units of complete tiles.
 */
template<ducks::st::all ST>
__device__ static inline void load_async(ST &dst, void const* const src_tma_map, barrier& bar, int tile_row_idx, int tile_col_idx=0) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (dst.rows);
        int32_t crd2 = tile_col_idx * (dst.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2)
            : "memory"
        );
    }
}

namespace cluster {

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile, across a threadblock cluster
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The barrier used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row index of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column index of the requested tile. This is in units of complete tiles.
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 */
template<ducks::st::all ST>
__device__ static inline void load_async(ST &dst, void const* const src_tma_map, barrier& bar, int tile_row_idx, int tile_col_idx, uint16_t cluster_mask) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));

        int32_t crd0 = 0;
        int32_t crd1 = tile_row_idx * (dst.rows);
        int32_t crd2 = tile_col_idx * (dst.cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));

        asm volatile (
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5}], [%2], %6;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "h"(cluster_mask)
            : "memory"
        );
    }
}

} // namespace cluster
} // namespace tma
} // namespace kittens