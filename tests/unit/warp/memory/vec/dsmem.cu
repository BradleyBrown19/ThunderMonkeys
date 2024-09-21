#include "dsmem.cuh"
#include <cooperative_groups.h>

#ifdef TEST_WARP_MEMORY_VEC_DSMEM

template<typename T>
struct test_dsmem_vec { // load with dsmem, write out normally
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && (S*kittens::TILE_DIM*sizeof(dtype))%128==0>; // note the 128 byte multiple requirement
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "dsmem_vec_transfer_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "dsmem_vec_transfer_gmem=half" :
                                                                                         "dsmem_vec_transfer_gmem=float";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < S*16; j++) {
                o_ref[i*S*16 + j] = i_ref[((i+1)%4)*S*16 + j];
            }
        }
    }
    template<int S, int NW>
    __device__ static void device_func(const dtype *input, dtype *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st<dtype, S, S>> (&src_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>>();
        kittens::row_vec<kittens::st<dtype, S, S>> (&dst_vec) = al.allocate<kittens::row_vec<kittens::st<dtype, S, S>>>();

        __shared__ kittens::barrier dsmem_barrier;
        kittens::load(src_vec, input + blockIdx.x*src_vec.length);

        kittens::init_barrier(dsmem_barrier, 0, 1);
        kittens::tma::expect<typeof(dst_vec)>(dsmem_barrier);

        auto cluster = cooperative_groups::this_cluster();
        cluster.sync(); // ensure everyone has initialized their barrier

        kittens::tma::cluster::store_async(dst_vec, src_vec, 4, (blockIdx.x+3)%4, dsmem_barrier);

        kittens::wait(dsmem_barrier, 0);

        kittens::store(output + blockIdx.x*dst_vec.length, dst_vec);
    }
};

template<typename Ker, typename T, int S, int NW, typename... args>
static __global__ __cluster_dims__(4, 1, 1) void dsmem_global_wrapper_1d(const T *input, T *output) {
    Ker::template device_func<S, NW, args...>(input, output);
}
template<typename test, int S, int NUM_WORKERS, typename... args>
struct dsmem_wrapper_1d {
    using dtype = gmem_dtype<test>;
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<S, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = S*16 * 4; // 4 for additional dsmem cluster dimension
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<dtype, initializers::ARANGE>(&d_i, &d_o, i_ref, o_ref);
            // run kernel
            cudaFuncSetAttribute(
                dsmem_global_wrapper_1d<test, dtype, S, NUM_WORKERS, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            dsmem_global_wrapper_1d<test, dtype, S, NUM_WORKERS, args...><<<4, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o);
            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, S*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_S, int NW, typename... args>
using dsmem_sweep_size_1d = loop_s<dsmem_wrapper_1d, test, MAX_S, NW, MAX_S, args...>;
template<template<typename> typename test, int MAX_S=8, int NUM_WORKERS=1, typename... args>
struct dsmem_sweep_gmem_type_1d {
    static void run(test_data &results) {
        dsmem_sweep_size_1d<test<float>, MAX_S, NUM_WORKERS, args...>::run(results);
        dsmem_sweep_size_1d<test<kittens::bf16>, MAX_S, NUM_WORKERS, args...>::run(results);
        dsmem_sweep_size_1d<test<kittens::half>, MAX_S, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, int MAX_S=8, typename... args> using dsmem_sweep_gmem_type_1d_warp = dsmem_sweep_gmem_type_1d<test, MAX_S, 1, args...>;

void warp::memory::vec::dsmem::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/dsmem tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    dsmem_sweep_gmem_type_1d_warp<test_dsmem_vec, SIZE>::run(results);
}

#endif