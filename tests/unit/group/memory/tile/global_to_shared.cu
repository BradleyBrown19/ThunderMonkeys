#include "global_to_shared.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename T>
struct group_shared_load_store {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 &&
        W*H<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_loadstore_gmem=half" :
                                                                                         "group_shared_loadstore_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW> __device__ static void device_func(const dtype *input, dtype *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st<dtype, H, W> &shared_tile = al.allocate<kittens::st<dtype, H, W>>();
        G::load(shared_tile, input, W*16);
        G::store(output, shared_tile, W*16);
    }
};
template<typename T>
struct group_shared_load_store_async {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 &&
        H*W<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_loadstore_async_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_loadstore_async_gmem=half" :
                                                                                         "group_shared_loadstore_async_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW> __device__ static void device_func(const dtype *input, dtype *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 

        auto block = cooperative_groups::this_thread_block();
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, block.size());}
        block.sync();
        
        kittens::st<dtype, H, W> &shared_tile = al.allocate<kittens::st<dtype, H, W>>();

        block.sync();
        G::load_async(shared_tile, input, W*16, barrier);
        barrier.arrive_and_wait();

        G::store_async(output, shared_tile, W*16, barrier);
        barrier.arrive_and_wait();
    }
};

void group::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<group_shared_load_store<float>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store<float>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store<float>, SIZE, 4, 12>::run(results);
    sweep_size_2d<group_shared_load_store_async<float>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store_async<float>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store_async<float>, SIZE, 4, 12>::run(results);

    sweep_size_2d<group_shared_load_store<kittens::bf16>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::bf16>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::bf16>, SIZE, 4, 12>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::bf16>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::bf16>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::bf16>, SIZE, 4, 12>::run(results);

    sweep_size_2d<group_shared_load_store<kittens::half>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::half>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::half>, SIZE, 4, 12>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::half>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::half>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::half>, SIZE, 4, 12>::run(results);
}

#endif