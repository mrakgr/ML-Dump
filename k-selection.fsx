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

let row_size = 1024
let column_size = 250
let dmat = createRandomUniformMatrix row_size column_size 3.0f
let output = createEmptyMatrixLike dmat

(*
An extremely fast activation function. This one trully deserves the name
of k-selection. Because row_size is determined statically and is a multiple of 32, I've 
been able to unroll all the loops and store the variables into registers.

1.19 seconds per 10k iterations. Is only 3x slower than map and roughly 50x faster
than sort.

On Maxwell cards having the small block size of 32 is very efficient.
http://arxiv.org/abs/1312.5663
*)
type sparsePiecewiseLinearActivationModule(target, num_rows, num_splits) =
    inherit GPUModule(target)

    let grid_size = 384
    let block_size = 32

    new (num_rows, num_splits) = new sparsePiecewiseLinearActivationModule(GPUModuleTarget.Worker(worker), num_rows, num_splits)
    /// Default number of splits=20
    new num_rows = new sparsePiecewiseLinearActivationModule(GPUModuleTarget.Worker(worker), num_rows, 20)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (num_cols:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (k: int) =
        let inline butterflyWarpSum (value: int) = 
            let v1 = value + __shfl_xor value 16 32
            let v2 = v1 + __shfl_xor v1 8 32
            let v3 = v2 + __shfl_xor v2 4 32
            let v4 = v3 + __shfl_xor v3 2 32
            v4 + __shfl_xor v4 1 32

        let inline butterflyWarpMinMax (value:float32) = 
            let v = __shfl_xor value 16 32
            let minval1 = min v value
            let maxval1 = max v value
            let v1 = __shfl_xor v 8 32
            let minval2 = min v1 minval1
            let maxval2 = max v1 maxval1
            let v2 = __shfl_xor v1 4 32
            let minval3 = min v2 minval2
            let maxval3 = max v2 maxval2
            let v3 = __shfl_xor v2 2 32
            let minval4 = min v3 minval3
            let maxval4 = max v3 maxval3
            let v4 = __shfl_xor v3 1 32
            let minval5 = min v4 minval4
            let maxval5 = max v4 maxval4
            minval5, maxval5

        let num_vars = num_rows/__warp_size()
        let vars = __local__.Array(num_vars)
        // Point block_start to where the column starts in the array.
        let mutable col = blockIdx.x

        while col < num_cols do
            // i is the variable index
            // Store the variables into registers.
            // The reason the num_rows is static and multiple of 32 is so I
            // can unroll this loop and guarantee that the registers will be used
            // instead of spilled to global memory.
            let mutable column_min, column_max = System.Single.MaxValue, System.Single.MinValue
            __unroll()
            for i=0 to num_vars-1 do
                // idx is the absolute index in the array
                let idx = threadIdx.x + i*32 + col * num_rows
                vars.[i] <- x.[idx]
                let tmin, tmax = butterflyWarpMinMax vars.[i]
                column_min <- min tmin column_min
                column_max <- max tmax column_max

            // Split the range in the direction of k num_splits times for 
            // 2^num_splits precision.
            __unroll()
            for iters=1 to num_splits do
                let guess = (column_min+column_max)/2.0f
                let mutable count = 0
                __unroll()
                for i=0 to num_vars-1 do
                    let c = if vars.[i] >= guess then 1 else 0
                    count <- count+c
                count <- butterflyWarpSum count
                if count > k then column_min <- guess 
                else if count < k then column_max <- guess

            let threshold = (column_min+column_max)/2.0f
            __unroll()
            for i=0 to num_vars-1 do
                // idx is the absolute index in the array
                let idx = threadIdx.x + i*32 + col * num_rows
                
                // Let the function activate if it is above the threshold.
                y.[idx] <- if vars.[i] >= threshold then vars.[i] else 0.0f

            col <- col + gridDim.x

    member this.Apply((dmat: dM), (output: dMatrix<float32>), k) =
        if dmat.num_rows % 32 <> 0 then failwith "dmat.dArray.num_rows have to be a multiple of 32 in sparsePiecewiseLinearActivationModule"
        if dmat.dArray.Length <> output.dArray.Length then failwith "dmat.dArray.Length <> output.dArray.Length in sparsePiecewiseLinearActivationModule"
        let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
        this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr output.dArray.Ptr k
        output

    member this.Apply((dmat: dM), k) =
        if dmat.num_rows % 32 <> 0 then failwith "dmat.dArray.num_rows have to be a multiple of 32 in sparsePiecewiseLinearActivationModule"
        let output = createEmptyMatrixLike dmat
        let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
        this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr output.dArray.Ptr k
        output

let m = new sparsePiecewiseLinearActivationModule(row_size)
m.GPUForceLoad()

#time
for i=1 to 10000 do
    m.Apply(dmat,output,25) |> ignore
    worker.Synchronize()
#time

let t = output.dArray.Gather()
t |> Array.sum