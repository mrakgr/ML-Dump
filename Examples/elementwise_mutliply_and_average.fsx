// A module for elementwise multiplication and then averaging

// It is necessary for the LSTM peepholes because they are n by 1 matrices
// while cell state and errors are n by m matrices. I need to average them 
// to get the former

#load "utils.fsx"
open Utils.Utils

open System
open System.IO

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open Alea.CUDA.CULib.CUBLASInterop
open Alea.CUDA.CULib.CUDNNInterop
open Alea.CUDA.IL
open Alea.CUDA.Unbound.Rng
open Alea.CUDA.Unbound
open FSharp.Quotations

type elementwiseMultiplyAndAverageModule(target, slice_size, groups_per_slice) =
    inherit GPUModule(target)

    /// Default number of splits=20
    new () = 
        let slice_size = 32
        let groups_per_slice = 8
        new elementwiseMultiplyAndAverageModule(GPUModuleTarget.Worker(worker),slice_size,groups_per_slice)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (num_rows:int) (num_cols:int) alpha (x:deviceptr<float32>) (y:deviceptr<float32>) beta (z:deviceptr<float32>) =
        // Multiple groups are assigned to a single slice.
        // blockDim.x = number of threads in a group assigned to a row slice = slice_size
        // blockDim.y = number of groups assigned to a single slice = groups_per_slice

        // This can be a bit confusing as threadIdx.x refers to the thread in a particular block
        // while blockIdx.x refers to the block (of a grid)
        let mutable row = blockIdx.x*blockDim.x+threadIdx.x
        
        // threadIdx.y = the index of a group
        let mutable col = threadIdx.y

        // Shared memory for the block
        let mem = __shared__.Array(slice_size)
        let mem_ptr = __array_to_ptr mem

        while row < num_rows do

            // If the group is zero then set the shared memory to zero.
            if threadIdx.y = 0 then mem.[threadIdx.x] <- 0.0f
            __syncthreads()

            let mutable sum = 0.0f

            while col < num_cols do
                let idx = row + col*num_rows

                // Elementwise multiplication. The values are gathered into shared memory using atomics afterwards.
                sum <- sum + y.[idx]*x.[idx]

                // blockDim.y = number of groups assigned to a single slice
                col <- col + blockDim.y

            // blockDim.x = number of threads in a group assigned to a row slice
            // gridDim.x = number slices
            __atomic_add (mem_ptr+threadIdx.x) sum |> ignore

            __syncthreads()
            if threadIdx.y = 0 then 
                let divisor = float32 num_cols
                z.[row] <- alpha*mem.[threadIdx.x]/divisor + beta*z.[row]
            __syncthreads()     
                   
            // Strided looping
            row <- row + blockDim.x*gridDim.x
        

    member this.ElementwiseMultiplyAndAverage(alpha, x: dM, y: dM, beta, z: dM) =
        if x.num_rows <> y.num_rows || y.num_rows <> z.num_rows then failwith "x.num_rows <> y.num_rows || y.num_rows <> z.num_rows in elementwiseMultiplyAndAverageModule"
        if z.num_cols <> 1 then failwith "z should have num_cols = 1 in elementwiseMultiplyAndAverageModule"
        if x.num_cols <> y.num_cols || x.dArray.Length <> y.dArray.Length 
        then failwith "The sizes of x are not equivalent to y in elementwiseMultiplyAndAverageModule"

        let slice_size = min slice_size x.num_rows
        let groups_per_slice = min groups_per_slice y.num_cols

        let dims_block = dim3(slice_size,groups_per_slice)
        let dims_grid = dim3(divup x.num_rows slice_size)
        let lp = LaunchParam(dims_grid,dims_block)

        this.GPULaunch <@ this.Kernel @> lp y.num_rows y.num_cols alpha x.dArray.Ptr y.dArray.Ptr beta z.dArray.Ptr
        z

    member this.ElementwiseMultiplyAndAverage(alpha, x: dM, y: dM, beta) =
        let z = createEmptyMatrix x.num_rows 1
        setModule.Apply(0.0f,z,z) |> ignore
        this.ElementwiseMultiplyAndAverage(alpha,x,y,beta,z)

let rng = System.Random()
let rows = 200
let cols = 350
let matrix1 = [|for i=1 to rows*cols do yield rng.NextDouble() |> float32|]
let matrix2 = [|for i=1 to rows*cols do yield rng.NextDouble() |> float32|]
let total = [|for i=0 to rows*cols-1 do yield matrix1.[i]*matrix2.[i]|]
let averaged = 
    [|for i=0 to rows-1 do 
        let mutable sum = 0.0f
        for k=0 to cols-1 do
            sum <- sum + total.[k*rows+i]
        yield sum|]

let createMatrix rows cols array =
    {num_rows=rows;num_cols=cols; dArray=array}:dM

let x = createMatrix rows cols (worker.Malloc(matrix1))
let y = createMatrix rows cols (worker.Malloc(matrix2))
let z = createMatrix rows 1 (worker.Malloc<float32>(rows))

let averager = new elementwiseMultiplyAndAverageModule(GPUModuleTarget.Worker(worker),32,8)
averager.GPUForceLoad()

let t = averager.ElementwiseMultiplyAndAverage(1.0f,x,y,0.0f,z)
let test = t.dArray.Gather()

let r = Array.map2 (fun a b -> abs(a-b)) test averaged

r |> Array.max

// ~1.2s on GTX 970 with a slice_size=32 and groups_per_slice=8
// Gives roughly similar results for a wide variety of parameters.
#time
for i=1 to 10000 do
    averager.ElementwiseMultiplyAndAverage(x,y,z) |> ignore
    worker.Synchronize()
#time
