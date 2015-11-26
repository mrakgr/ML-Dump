#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.2.1.2.3274\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.IL.2.1.2.3274\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.Unbound.2.1.2.3274\lib\net40\"
#r @"Alea.CUDA.Unbound.dll"
#r @"Alea.CUDA.IL.dll"
#r @"Alea.CUDA.dll"
#r "System.Configuration.dll"

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open Alea.CUDA.IL
open FSharp.Quotations

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- __SOURCE_DIRECTORY__ + @"\..\..\..\packages\Alea.CUDA.2.0.3057\private"
Alea.CUDA.Settings.Instance.Resource.Path <- __SOURCE_DIRECTORY__ + @"\..\..\..\release"

let worker = Worker.Default

[<Record>]
type cost_ind ={
    mutable cost : float32
    mutable ind : int
    }

/// This is specially made sequential kernel made to reduce 10xm matrix of Mnist digists.
/// It finds the maximum of a column along with its index.
[<ReflectedDefinition>]
let rowReduceMnist (m:int) (x:deviceptr<float32>) (y:deviceptr<cost_ind>) =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    let start = tid*10
    let mutable cost, row_index = Single.MinValue, -1
    if tid < m then
        for i=0 to 9 do
            let new_cost = x.[start+i]
            if cost < new_cost then
                cost <- new_cost
                row_index <- i
        y.[tid].cost <- cost
        y.[tid].ind <- row_index

let testrowReduceMnist() =
    let columns = 5

    let rng = Random(1)
    let x = Array.init (10*columns) (fun _ -> float32(rng.NextDouble())) 
    printfn "%A" (x)
    use dx = worker.Malloc(x)
    use dy = worker.Malloc<cost_ind>(columns)

    let blockSize = 256
    let gridSize = divup columns blockSize
    let lp = LaunchParam(gridSize, blockSize)

    worker.Launch <@rowReduceMnist@> lp columns dx.Ptr dy.Ptr

    printfn "%A" (dy.Gather())

//testrowReduceMnist()