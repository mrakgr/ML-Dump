// A module for elementwise multiplication using broadcasting.
// It is necessary for the LSTM peepholes and error derivatives.

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

/// Note: The utils.fsx version is modified so it adds to z.
type broadcastingMultiplicationModule(target, slice_size, groups_per_slice) =
    inherit GPUModule(target)

    /// Default number of splits=20
    new () = 
        let slice_size = 32
        let groups_per_slice = 8
        new broadcastingMultiplicationModule(GPUModuleTarget.Worker(worker),slice_size,groups_per_slice)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (num_rows:int) (num_cols:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (z:deviceptr<float32>) =
        // Multiple groups are assigned to a single slice.
        // blockDim.x = number of threads in a group assigned to a row slice
        // blockDim.y = number of groups assigned to a single slice

        // This can be a bit confusing as threadIdx.x refers to the thread in a particular block
        // while blockIdx.x refers to the block (of a grid)
        let mutable row = blockIdx.x*blockDim.x+threadIdx.x
        
        // threadIdx.y = the index of a group
        let mutable col = threadIdx.y

        while row < num_rows do
            // Each thread loads its own element
            let elem = x.[row]
            while col < num_cols do
                let idx = row + col*num_rows

                // Elementwise multiplication with broadcasting.
                z.[idx] <- y.[idx]*elem

                // blockDim.y = number of groups assigned to a single slice
                col <- col + blockDim.y

            // blockDim.x = number of threads in a group assigned to a row slice
            // gridDim.x = number slices
            
            // Strided looping
            row <- row + blockDim.x*gridDim.x
        

    member this.BroadcastMultiply(x: dM, y: dM, z: dM) =
        if x.num_rows <> y.num_rows then failwith "x.num_rows <> y.num_rows in broadcastingMultiplicationModule"
        if x.num_cols <> 1 then failwith "x should have num_cols = 1 in broadcastingMultiplicationModule"
        if z.num_cols <> y.num_cols || z.num_rows <> y.num_rows || z.dArray.Length <> y.dArray.Length 
        then failwith "The sizes of z are not equivalent to y in broadcastingMultiplicationModule"

        let slice_size = min slice_size x.num_rows
        let groups_per_slice = min groups_per_slice y.num_cols

        let dims_block = dim3(slice_size,groups_per_slice)
        let dims_grid = dim3(divup x.num_rows slice_size)
        let lp = LaunchParam(dims_grid,dims_block)

        this.GPULaunch <@ this.Kernel @> lp y.num_rows y.num_cols x.dArray.Ptr y.dArray.Ptr z.dArray.Ptr
        z

    member this.BroadcastMultiply(x: dM, y: dM) =
        let z = createEmptyMatrixLike y
        this.BroadcastMultiply(x,y,z)

let rng = System.Random()
let rows = 200
let cols = 350
let multiplier = [|for i=1 to rows do yield rng.NextDouble() |> float32|]
let matrix = [|for i=1 to rows*cols do yield rng.NextDouble() |> float32|]
let total = [|for i=0 to rows*cols-1 do yield multiplier.[i % rows]*matrix.[i]|]

let createMatrix rows cols array =
    {num_rows=rows;num_cols=cols; dArray=array}:dM

let x = createMatrix rows 1 (worker.Malloc(multiplier))
let y = createMatrix rows cols (worker.Malloc(matrix))
let broad = new broadcastingMultiplicationModule(GPUModuleTarget.Worker(worker),32,8)
broad.GPUForceLoad()

let z = broad.BroadcastMultiply(x,y)
let test = z.dArray.Gather()

let r = Array.map2 (fun a b -> abs(a-b)) test total

r |> Array.max

// ~1.2s on GTX 970 with a slice_size=32 and groups_per_slice=8
// Gives roughly similar results for a wide variety of parameters.
#time
for i=1 to 10000 do
    broad.BroadcastMultiply(x,y,z) |> ignore
    worker.Synchronize()
#time