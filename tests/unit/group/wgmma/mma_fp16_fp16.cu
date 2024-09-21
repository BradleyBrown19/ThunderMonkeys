#include "mma_fp16_fp16.cuh"

#ifdef TEST_GROUP_WGMMA_MMA_FP16_FP16

struct test_mma_AB_fp16_fp16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>;
    static inline const std::string test_identifier = "wgmma_mma_AB_fp16_fp16";
    template<int H, int W, int NW, typename _K>
     __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i*16*K + k]*i_ref[(256*H*K) + k*16*W + j];
                }
                o_ref[i*16*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K>
    __device__ static void device_func(const kittens::half *input, kittens::half *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_hf<H, K> &a = al.allocate<kittens::st_hf<H, K>>();
        kittens::st_hf<K, W> &b = al.allocate<kittens::st_hf<K, W>>();
        kittens::rt_hf<1, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, K*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, W*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_AB(c, a, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
    }
};
struct test_mma_ABt_fp16_fp16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "wgmma_mma_ABt_fp16_fp16";
    template<int H, int W, int NW, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i*K*16+k]*i_ref[256*K*H + j*K*16+k];
                }
                o_ref[i*W*16+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K>
    __device__ static void device_func(const kittens::half *input, kittens::half *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_hf<H, K> &a = al.allocate<kittens::st_hf<H, K>>();
        kittens::st_hf<W, K> &b = al.allocate<kittens::st_hf<W, K>>();
        kittens::rt_hf<1, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, K*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, K*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_ABt(c, a, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
    }
};
struct test_mma_AtB_fp16_fp16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "wgmma_mma_AtB_fp16_fp16";
    template<int H, int W, int NW, typename _K>
     __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[k*16*H + i]*i_ref[(256*H*K) + k*16*W + j];
                }
                o_ref[i*16*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K>
    __device__ static void device_func(const kittens::half *input, kittens::half *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_hf<K, H> &a = al.allocate<kittens::st_hf<K, H>>();
        kittens::st_hf<K, W> &b = al.allocate<kittens::st_hf<K, W>>();
        kittens::rt_hf<1, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, H*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, W*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_AtB(c, a, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
    }
};
struct test_mma_AtBt_fp16_fp16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "wgmma_mma_AtBt_fp16_fp16";
    template<int H, int W, int NW, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[k*16*H + i]*i_ref[256*K*H + j*K*16+k];
                }
                o_ref[i*W*16+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K>
    __device__ static void device_func(const kittens::half *input, kittens::half *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_hf<K, H> &a = al.allocate<kittens::st_hf<K, H>>();
        kittens::st_hf<W, K> &b = al.allocate<kittens::st_hf<W, K>>();
        kittens::rt_hf<1, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, H*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, K*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_AtBt(c, a, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
    }
};

// register+shared version
struct reg_test_mma_AB_fp16_fp16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H%4==0 && (2*W*H+W*K::value+H*K::value)<=256>;
    static inline const std::string test_identifier = "wgmma_reg_mma_AB_fp16_fp16";
    template<int H, int W, int NW, typename _K>
     __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i*16*K + k]*i_ref[(256*H*K) + k*16*W + j];
                }
                o_ref[i*16*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K>
    __device__ static void device_func(const kittens::half *input, kittens::half *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_hf<H, K> &a = al.allocate<kittens::st_hf<H, K>>();
        kittens::st_hf<K, W> &b = al.allocate<kittens::st_hf<K, W>>();
        kittens::rt_hf<H/4, K> a_reg;
        kittens::rt_hf<H/4, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, K*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, W*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::load(a_reg, a);
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_AB(c, a_reg, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
    }
};
struct reg_test_mma_ABt_fp16_fp16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H%4==0 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "wgmma_reg_mma_ABt_fp16_fp16";
    template<int H, int W, int NW, typename _K>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += i_ref[i*K*16+k]*i_ref[256*K*H + j*K*16+k];
                }
                o_ref[i*W*16+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, typename _K>
    __device__ static void device_func(const kittens::half *input, kittens::half *output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_hf<H, K> &a = al.allocate<kittens::st_hf<H, K>>();
        kittens::st_hf<W, K> &b = al.allocate<kittens::st_hf<W, K>>();
        kittens::rt_hf<H/4, K> a_reg;
        kittens::rt_hf<H/4, W> c;
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, kittens::WARPGROUP_THREADS);}
        __syncthreads();
        kittens::warpgroup::load_async(a, input, K*16, barrier);
        kittens::warpgroup::load_async(b, input+a.num_elements, K*16, barrier);
        barrier.arrive_and_wait();
        kittens::warpgroup::load(a_reg, a);
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_ABt(c, a_reg, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(output, c, W*16);
    }
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            half *d_i, *d_o;
            std::vector<float> i_ref((H+W)*K*256);
            std::vector<float> o_ref(H*W*256);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_2d<test, kittens::half, H, W, NUM_WORKERS, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_2d<test, kittens::half, H, W, NUM_WORKERS, _K, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, _K, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16, 0.02); // wgmma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int H, int MAX_W, int NUM_WORKERS=1, typename... args> using mma_sweep_width = loop_w<mma_wrapper_2d, test, H, MAX_W, NUM_WORKERS, H, MAX_W, args...>;
template<typename test, int MAX_W, typename... args> using mma_sweep_width_warpgroup = mma_sweep_width<test, 4, MAX_W, 4, args...>;
template<typename test, int MAX_W, typename... args> using mma_sweep_width_warpgroup_doubleheight = mma_sweep_width<test, 8, MAX_W, 4, args...>;

// If 1 and 3 work, the others likely will too.
using I1_t = std::integral_constant<int, 1>;
using I3_t = std::integral_constant<int, 3>;
using I5_t = std::integral_constant<int, 5>;
void group::wgmma::mma_fp16_fp16::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warpgroup/wgmma/mma_fp16_fp16 tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2 :
                         INTENSITY_2 ? 4 :
                         INTENSITY_3 ? 8 :
                         INTENSITY_4 ? 16 : -1;

    // shared+shared->reg

    mma_sweep_width_warpgroup<test_mma_AB_fp16_fp16,   SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_ABt_fp16_fp16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_AtB_fp16_fp16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_AtBt_fp16_fp16, SIZE, I1_t>::run(results);

    mma_sweep_width_warpgroup<test_mma_AB_fp16_fp16,   SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_ABt_fp16_fp16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_AtB_fp16_fp16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_AtBt_fp16_fp16, SIZE, I3_t>::run(results);

    mma_sweep_width_warpgroup<test_mma_AB_fp16_fp16,   SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_ABt_fp16_fp16,  SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_AtB_fp16_fp16,  SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup<test_mma_AtBt_fp16_fp16, SIZE, I5_t>::run(results);

    // register+shared->reg

    mma_sweep_width_warpgroup<reg_test_mma_AB_fp16_fp16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<reg_test_mma_ABt_fp16_fp16, SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_test_mma_AB_fp16_fp16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_test_mma_ABt_fp16_fp16, SIZE, I1_t>::run(results);

    mma_sweep_width_warpgroup<reg_test_mma_AB_fp16_fp16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<reg_test_mma_ABt_fp16_fp16, SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_test_mma_AB_fp16_fp16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_test_mma_ABt_fp16_fp16, SIZE, I3_t>::run(results);

    mma_sweep_width_warpgroup<reg_test_mma_AB_fp16_fp16,  SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup<reg_test_mma_ABt_fp16_fp16, SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_test_mma_AB_fp16_fp16,  SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_test_mma_ABt_fp16_fp16, SIZE, I5_t>::run(results);
}

#endif