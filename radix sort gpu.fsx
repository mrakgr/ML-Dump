// Radix sort is unacceptably slow.

#load "utils.fsx"
open Utils.Utils

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open Alea.CUDA.CULib.CUBLASInterop
open Alea.CUDA.CULib.CUDNNInterop
open Alea.CUDA.IL
open Alea.CUDA.Unbound.Rng
open Alea.CUDA.Unbound
open FSharp.Quotations



//let weights = createRandomUniformMatrix 32 1 10.0f
//let output = createEmptyMatrixLike weights
let rng = System.Random()
let row_size = 1024
let column_size = 250
let random_array = Array.init (row_size*column_size) (fun _ -> rng.Next(1,41))
let weights = {num_rows = row_size; num_cols = column_size; dArray = worker.Malloc(random_array)}
let output = {num_rows = row_size; num_cols = column_size; dArray = worker.Malloc<int>(random_array.Length)}

/// The second version of radix sort that works on ints.
/// It works even worse than the previous version.
type radixBlockSortModule(target, shared_memory_size) =
    inherit GPUModule(target)

    let block_size = 128
    let blockScanner = BlockScan.Default<int>(dim3(block_size),worker.Device.Arch)

    new shared_memory_size =
        new radixBlockSortModule(GPUModuleTarget.Worker(worker), shared_memory_size)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (num_rows:int) (num_cols:int) (x:deviceptr<int>) (y:deviceptr<int>) =
        let temp_storage = blockScanner.TempStorage.AllocateShared()
        let shared_ar = __shared__.Array<int>(shared_memory_size)
        let shared_hist = __shared__.Array<int>((divup shared_memory_size (__warp_size()/2))+1)

        let variables = __local__.Array (divup shared_memory_size block_size)

        let mutable bid = blockIdx.x

        while bid < num_cols do
            // Load the columns into registers.
            let mutable tid = threadIdx.x
            while tid < num_rows do
                let var = x.[tid+bid*num_rows]
                variables.[tid/blockDim.x] <- var

                tid <- tid + blockDim.x

            __unroll()
            for i=0 to 31 do

                // Histogram calculations for each bit for each column.
                tid <- threadIdx.x
                while tid < num_rows do
                    let count = tid/blockDim.x
                    let var = variables.[count]
                    let mask = 1 <<< i
                    let bit = var &&& mask

                    let hist_ind = if bit <> 0 then 1 else 0

                    let positive_warp_threads = int (__ballot hist_ind)
                    let positive_warp_threads_count = __nv_popc positive_warp_threads

                    if __laneid() = 0 then 
                        shared_hist.[tid / __warp_size()+1] <- __warp_size() - positive_warp_threads_count
                        shared_hist.[tid / __warp_size() + (divup num_rows (__warp_size()))+1] <- positive_warp_threads_count
                    
                        //printfn "positive_warp_threads_count = %i ... = %i" (positive_warp_threads_count) (__warp_size() - positive_warp_threads_count)
                    
                    tid <- tid + blockDim.x

                let shared_hist_length = (divup num_rows (__warp_size()))*2

                // Set the first value to 0.
                if threadIdx.x = 0 then shared_hist.[0] <- 0
                __syncthreads()

                let mutable stride = 1
            
                // Exclusive scan of the partial histogram values in shared memory.
                while stride < shared_hist_length do
                    tid <- threadIdx.x
                    while tid < num_rows do
                        if tid+stride < shared_hist_length then
                            shared_hist.[tid+stride] <- shared_hist.[tid+stride]+shared_hist.[tid]
                        tid <- tid + blockDim.x
                    stride <- stride*2

            
                // For each warp load the histogram and do the sorting according to
                // the radix bit. Compute the offsets based on the histogram.
                tid <- threadIdx.x
                while tid < num_rows do
                    let var = variables.[tid/blockDim.x]
                    let offset0 = shared_hist.[tid / __warp_size()]
                    let offset1 = shared_hist.[tid / __warp_size() + (divup num_rows (__warp_size()))]

                    //if __laneid() = 0 then printfn "offset0 = %i offset1 = %i" offset0 offset1

                    let mask = 1 <<< i
                    let bit = var &&& mask

                    let hist_ind = if bit <> 0 then 1 else 0
                    let positive_warp_threads = int (__ballot hist_ind)
                    let warp_threads = if bit <> 0 then positive_warp_threads else (~~~positive_warp_threads)
                    let behind_current_position = (1 <<< __laneid())-1
                    let threads_behind_current_position = behind_current_position &&& warp_threads
                    let offset = __nv_popc threads_behind_current_position
                    let true_offset = if bit <> 0 then offset+offset1 else offset+offset0

                    //printf "%i, %i; " true_offset var

                    // Store the variable at the correct offset.
                    shared_ar.[true_offset] <- var

                    tid <- tid + blockDim.x

                __syncthreads()
                // Load the variables back into registers from shared memory.
                // Then in the next iteration repeat the process.
                tid <- threadIdx.x
                while tid < num_rows do
                    let var = shared_ar.[tid]
                    variables.[tid/blockDim.x] <- var
                    tid <- tid + blockDim.x

            tid <- threadIdx.x
            while tid < num_rows do
                y.[tid+bid*num_rows] <- shared_ar.[tid]
                tid <- tid + blockDim.x

            bid <- bid+gridDim.x
            
    member this.Apply((num_rows:int), (num_cols:int), (x:deviceptr<int>), (y:deviceptr<int>)) =
        if num_rows > shared_memory_size then failwith "The rows exceed shared memory size in radixBlockSortModule"
        let lp = LaunchParam(min num_cols 12, block_size)
        this.GPUWorker.ProfilerStart()
        this.GPULaunch <@ this.Kernel @> lp num_rows num_cols x y
        this.GPUWorker.Synchronize()
        this.GPUWorker.ProfilerStop()

    member this.Apply (x: dMatrix<int>, y: dMatrix<int>) =
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in radixBlockSortModule"
        this.Apply(x.num_rows, x.num_cols, x.dArray.Ptr, y.dArray.Ptr)

let sortModule = new radixBlockSortModule(row_size)
sortModule.GPUForceLoad()

#time
for i=1 to 1 do
    sortModule.Apply(weights,output)
    worker.Synchronize()
#time

//let hweights = weights.dArray.Gather()
//let houtput = output.dArray.Gather()

//let t1 = hweights |> Array.sum 
//let t2 = houtput |> Array.sum

//t1 = t2

(*
/// The first version of radix sort that works on ints.
/// Slow as a turd. For a 1024x250 matrix it takes 5.5 secs/1000 iterations to 
/// sort it.
type radixBlockSortModule(target, shared_memory_size) =
    inherit GPUModule(target)

    let block_size = 256
    let ommit_mask = (1 <<< 8)-1
    let blockScanner = BlockScan.Default<int>(dim3(block_size),worker.Device.Arch)

    new shared_memory_size =
        new radixBlockSortModule(GPUModuleTarget.Worker(worker), shared_memory_size)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (num_rows:int) (num_cols:int) (x:deviceptr<int>) (y:deviceptr<int>) =
        let ind = threadIdx.x + blockIdx.x*blockDim.x

        let matching_elem = threadIdx.x

        let temp_storage = blockScanner.TempStorage.AllocateShared()
        let shared1 = __shared__.Array<int>(shared_memory_size)
        let shared2 = __shared__.Array<int>(shared_memory_size)

        let mutable shared_ptr1 = shared1
        let mutable shared_ptr2 = shared2

        let hist = [|0;0;0;0|]

        for i=0 to num_rows-1 do
            let elem = x.[blockIdx.x*num_rows+i]
            shared1.[i] <- elem

            hist.[0] <- hist.[0] + if (elem &&& ommit_mask) = matching_elem then 1 else 0
            hist.[1] <- hist.[1] + if ((elem >>> 8) &&& ommit_mask) = matching_elem then 1 else 0
            hist.[2] <- hist.[2] + if ((elem >>> 16) &&& ommit_mask) = matching_elem then 1 else 0
            hist.[3] <- hist.[3] + if ((elem >>> 24) &&& ommit_mask) = matching_elem then 1 else 0

        hist.[0] <- blockScanner.ExclusiveScan temp_storage  (fun a b -> a+b) 0 hist.[0]
        __syncthreads()
        hist.[1] <- blockScanner.ExclusiveScan temp_storage  (fun a b -> a+b) 0 hist.[1]
        __syncthreads()
        hist.[2] <- blockScanner.ExclusiveScan temp_storage  (fun a b -> a+b) 0 hist.[2]
        __syncthreads()
        hist.[3] <- blockScanner.ExclusiveScan temp_storage  (fun a b -> a+b) 0 hist.[3]
        __syncthreads()

        let mutable j = 0
        while j <= 3 do
            for i=0 to num_rows-1 do
                let elem = shared_ptr1.[i] 
        
                if ((elem >>> (8*j)) &&& ommit_mask) = matching_elem then 
                    shared_ptr2.[hist.[j]] <- elem
                
                    hist.[j] <- hist.[j]+1

            // Switch pointers to the shared memory arrays.
            if j % 2 = 0 then
                shared_ptr1 <- shared2
                shared_ptr2 <- shared1
            else
                shared_ptr1 <- shared1
                shared_ptr2 <- shared2
            j <- j + 1

        let mutable tid = threadIdx.x
        while tid < num_rows do 
            y.[blockIdx.x*num_rows+tid] <- shared_ptr1.[tid]
            tid <- tid + blockDim.x
        ()
            
    member this.Apply((num_rows:int), (num_cols:int), (x:deviceptr<int>), (y:deviceptr<int>)) =
        let lp = LaunchParam(num_cols,block_size)
        this.GPULaunch <@ this.Kernel @> lp num_rows num_cols x y

    member this.Apply (x: dMatrix<int>, y: dMatrix<int>) =
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in radixBlockSortModule"
        this.Apply(x.num_rows, x.num_cols, x.dArray.Ptr, y.dArray.Ptr)
*)