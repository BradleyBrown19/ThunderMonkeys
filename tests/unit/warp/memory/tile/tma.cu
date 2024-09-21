#include "tma.cuh"

#ifdef TEST_WARP_MEMORY_TILE_TMA

template<typename T>
struct test_load { // load with TMA, write out normally
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_load_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_load_gmem=half" :
                                                                                         "tma_load_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<T, H, W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, H, W>, 2, 2>();
        
        __shared__ kittens::barrier smem_barrier; 
        kittens::init_barrier(smem_barrier, 0, 1);
        kittens::tma::expect<typeof(shared_tile[0][0]), 2, 2>(smem_barrier);
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::load_async(shared_tile[i][j], tma_desc_input, smem_barrier, i, j);
        }
        kittens::wait(smem_barrier, 0);

        kittens::store(output, shared_tile[0][0], 2*W*16);
        kittens::store(output + shared_tile[0][0].cols, shared_tile[0][1], 2*W*16);
        kittens::store(output + 2*shared_tile[0][0].num_elements, shared_tile[1][0], 2*W*16);
        kittens::store(output + 2*shared_tile[0][0].num_elements + shared_tile[0][0].cols, shared_tile[1][1], 2*W*16);
    }
};

template<typename T>
struct test_store { // load normally, store with TMA
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_gmem=half" :
                                                                                         "tma_store_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<T, H, W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, H, W>, 2, 2>();
        kittens::load(shared_tile[0][0], input, 2*W*16);
        kittens::load(shared_tile[0][1], input + shared_tile[0][0].cols, 2*W*16);
        kittens::load(shared_tile[1][0], input + 2*shared_tile[0][0].num_elements, 2*W*16);
        kittens::load(shared_tile[1][1], input + 2*shared_tile[0][0].num_elements + shared_tile[0][0].cols, 2*W*16);
        __syncwarp();
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::store_async(tma_desc_output, shared_tile[i][j], i, j);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename T>
struct test_store_add_reduce {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_add_reduce_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_add_reduce_gmem=half" :
                                                                                         "tma_store_add_reduce_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i]; 
        }
    }
    template<int H, int W, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<T, H, W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, H, W>, 2, 2>();
        kittens::load(shared_tile[0][0], input, 2*W*16);
        kittens::load(shared_tile[0][1], input + shared_tile[0][0].cols, 2*W*16);
        kittens::load(shared_tile[1][0], input + 2*shared_tile[0][0].num_elements, 2*W*16);
        kittens::load(shared_tile[1][1], input + 2*shared_tile[0][0].num_elements + shared_tile[0][0].cols, 2*W*16);
        __syncwarp();
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::store_add_async(tma_desc_output, shared_tile[i][j], i, j);
        }
        kittens::tma::store_commit_group();
        __syncwarp();
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::store_add_async(tma_desc_output, shared_tile[i][j], i, j);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename T>
struct test_store_min_reduce {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<!std::is_same_v<T, float> && NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_min_reduce_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_min_reduce_gmem=half" :
                                                                                         "tma_store_min_reduce_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::min(i_ref[i], o_ref[i]);
        }
    }
    template<int H, int W, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<T, H, W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, H, W>, 2, 2>();
        kittens::load(shared_tile[0][0], input, 2*W*16);
        kittens::load(shared_tile[0][1], input + shared_tile[0][0].cols, 2*W*16);
        kittens::load(shared_tile[1][0], input + 2*shared_tile[0][0].num_elements, 2*W*16);
        kittens::load(shared_tile[1][1], input + 2*shared_tile[0][0].num_elements + shared_tile[0][0].cols, 2*W*16);
        __syncwarp();
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::store_min_async(tma_desc_output, shared_tile[i][j], i, j);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename T>
struct test_store_max_reduce {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<!std::is_same_v<T, float> && NW == 1 && W*H*sizeof(dtype)*256*4<=kittens::MAX_SHARED_MEMORY-1024>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "tma_store_max_reduce_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "tma_store_max_reduce_gmem=half" :
                                                                                         "tma_store_max_reduce_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // i_ref is reduced onto output
        for (int i = 0; i < o_ref.size(); i++) {
            o_ref[i] = std::max(i_ref[i], o_ref[i]);
        }
    }
    template<int H, int W, int NW>
    __device__ static void device_func(const dtype *input, dtype *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st<T, H, W> (&shared_tile)[2][2] = al.allocate<kittens::st<T, H, W>, 2, 2>();
        kittens::load(shared_tile[0][0], input, 2*W*16);
        kittens::load(shared_tile[0][1], input + shared_tile[0][0].cols, 2*W*16);
        kittens::load(shared_tile[1][0], input + 2*shared_tile[0][0].num_elements, 2*W*16);
        kittens::load(shared_tile[1][1], input + 2*shared_tile[0][0].num_elements + shared_tile[0][0].cols, 2*W*16);
        __syncwarp();
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::store_max_async(tma_desc_output, shared_tile[i][j], i, j);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename Ker, typename T, int H, int W, int NW, typename... args>
static __global__ void tma_global_wrapper_2d(const T *input, T *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
    Ker::template device_func<H, W, NW, args...>(input, output, tma_desc_input, tma_desc_output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct tma_wrapper_2d {
    using dtype = gmem_dtype<test>;
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = H*W*256 * 2*2; // 2*2 for additional TMA dimensions
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);

            initialize<dtype, initializers::RANDOM>(&d_i, &d_o, i_ref, o_ref); 
            
            // initialize TMA descriptors
            CUtensorMap *i_desc = kittens::tma::allocate_and_create_tensor_map<kittens::st<dtype, H, W>>(d_i, 2, 2);
            CUtensorMap *o_desc = kittens::tma::allocate_and_create_tensor_map<kittens::st<dtype, H, W>>(d_o, 2, 2);
            // run kernel
            cudaFuncSetAttribute(
                tma_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tma_global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o, i_desc, o_desc);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, 2*W*16);
            cudaFree(i_desc);
            cudaFree(o_desc);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using tma_sweep_size_2d = loop_h<tma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct tma_sweep_gmem_type_2d {
    static void run(test_data &results) {
        tma_sweep_size_2d<test<float>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        tma_sweep_size_2d<test<kittens::bf16>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        tma_sweep_size_2d<test<kittens::half>, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, int MAX_H=8, int MAX_W=8, typename... args> using tma_sweep_gmem_type_2d_warp = tma_sweep_gmem_type_2d<test, MAX_H, MAX_W, 1, args...>;


void warp::memory::tile::tma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/tma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tma_sweep_gmem_type_2d<test_load,             SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store,            SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store_add_reduce, SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store_min_reduce, SIZE, SIZE>::run(results);
    tma_sweep_gmem_type_2d<test_store_max_reduce, SIZE, SIZE>::run(results);
}

#endif