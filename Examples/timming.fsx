//#load "utils.fsx"
//open Utils.Utils
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.2.1.2.3274\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.IL.2.1.2.3274\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.Unbound.2.1.2.3274\lib\net40\"
#r @"Alea.CUDA.Unbound.dll"
#r @"Alea.CUDA.IL.dll"
#r @"Alea.CUDA.dll"
#r "System.Configuration.dll"

open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open Alea.CUDA.Unbound
open Microsoft.FSharp.Quotations

type MapModule(target, op:Expr<float32 -> float32>) =
    inherit GPUModule(target)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (C:deviceptr<float32>) (A:deviceptr<float32>) (B:deviceptr<float32>) (n:int) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start
        while i < n do
            C.[i] <- __eval(op) A.[i] + __eval(op) B.[i]
            i <- i + stride

    member this.Apply(C:deviceptr<float32>, A:deviceptr<float32>, B:deviceptr<float32>, n:int) =
        let lp = LaunchParam(64, 256)
        this.GPULaunch <@ this.Kernel @> lp C A B n

let inline mapTemplate (op:Expr<'T -> 'T>) = cuda {
    let! kernel = 
        <@ fun (C:deviceptr<'T>) (A:deviceptr<'T>) (B:deviceptr<'T>) (n:int) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                C.[i] <- (%op) A.[i] + (%op) B.[i]
                i <- i + stride @>
        |> Compiler.DefineKernel

    return Entry(fun program ->
        let worker = program.Worker
        let kernel = program.Apply kernel
        let lp = LaunchParam(64, 256)

        let run C A B n =
            kernel.Launch lp C A B n

        run ) }

let test1 (worker:Worker) m n sync iters =
    let n = m * n
    use m = new MapModule(GPUModuleTarget.Worker(worker), <@ fun x -> x * 2.0f @>)
    let rng = System.Random(42)
    use A = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use B = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use C = worker.Malloc<float32>(n)
    let timer = System.Diagnostics.Stopwatch.StartNew()
    for i = 1 to iters do
        m.Apply(C.Ptr, A.Ptr, B.Ptr, n)
    if sync then worker.Synchronize()
    timer.Stop()
    printfn "%f ms / %d %s (no pre-load module)" timer.Elapsed.TotalMilliseconds iters (if sync then "sync" else "nosync")

let test2 (worker:Worker) m n sync iters =
    let n = m * n
    use m = new MapModule(GPUModuleTarget.Worker(worker), <@ fun x -> x * 2.0f @>)
    // we pre-load the module, this will JIT compile the GPU code
    m.GPUForceLoad()
    let rng = System.Random(42)
    use A = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use B = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use C = worker.Malloc<float32>(n)
    let timer = System.Diagnostics.Stopwatch.StartNew()
    for i = 1 to iters do
        m.Apply(C.Ptr, A.Ptr, B.Ptr, n)
    if sync then worker.Synchronize()
    timer.Stop()
    printfn "%f ms / %d %s (pre-loaded module)" timer.Elapsed.TotalMilliseconds iters (if sync then "sync" else "nosync")

let test3 (worker:Worker) m n sync iters =
    let n = m * n
    use m = new MapModule(GPUModuleTarget.Worker(worker), <@ fun x -> x * 2.0f @>)
    // we pre-load the module, this will JIT compile the GPU code
    m.GPUForceLoad()
    let rng = System.Random(42)
    use A = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use B = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use C = worker.Malloc<float32>(n)
    // since the worker is running in a background thread
    // each cuda api will switch to that thread
    // use eval() to avoid the many thread switching
    worker.Eval <| fun _ ->
        let timer = System.Diagnostics.Stopwatch.StartNew()
        for i = 1 to iters do
            m.Apply(C.Ptr, A.Ptr, B.Ptr, n)
        if sync then worker.Synchronize()
        timer.Stop()
        printfn "%f ms / %d %s (pre-loaded module + worker.eval)" timer.Elapsed.TotalMilliseconds iters (if sync then "sync" else "nosync")

let test4 (worker:Worker) m n sync iters =
    use program = worker.LoadProgram(mapTemplate <@ fun x -> x * 2.0f @>)
    let n = m * n
    let rng = System.Random(42)
    use A = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use B = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use C = worker.Malloc<float32>(n)
    let timer = System.Diagnostics.Stopwatch.StartNew()
    for i = 1 to iters do
        program.Run C.Ptr A.Ptr B.Ptr n
    if sync then worker.Synchronize()
    timer.Stop()
    printfn "%f ms / %d %s (template usage)" timer.Elapsed.TotalMilliseconds iters (if sync then "sync" else "nosync")

let test5 (worker:Worker) m n sync iters =
    use program = worker.LoadProgram(mapTemplate <@ fun x -> x * 2.0f @>)
    let n = m * n
    let rng = System.Random(42)
    use A = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use B = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use C = worker.Malloc<float32>(n)
    worker.Eval <| fun _ ->
        let timer = System.Diagnostics.Stopwatch.StartNew()
        for i = 1 to iters do
            program.Run C.Ptr A.Ptr B.Ptr n
        if sync then worker.Synchronize()
        timer.Stop()
        printfn "%f ms / %d %s (template usage + worker.Eval)" timer.Elapsed.TotalMilliseconds iters (if sync then "sync" else "nosync")

let test6 (worker:Worker) m n sync iters =
    use cublas = new CUBLAS(worker)
    let rng = System.Random(42)
    use dmat1 = worker.Malloc(Array.init (m * n) (fun _ -> rng.NextDouble() |> float32))
    use dmat2 = worker.Malloc(Array.init (m * n) (fun _ -> rng.NextDouble() |> float32))
    use dmatr = worker.Malloc<float32>(m * n)
    let timer = System.Diagnostics.Stopwatch.StartNew()
    for i = 1 to iters do
        cublas.Sgeam(cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, m, n, 2.0f, dmat1.Ptr, m, 2.0f, dmat2.Ptr, m, dmatr.Ptr, m)
    if sync then worker.Synchronize()
    timer.Stop()
    printfn "%f ms / %d %s (cublas)" timer.Elapsed.TotalMilliseconds iters (if sync then "sync" else "nosync")

let test7 (worker:Worker) m n sync iters =
    use cublas = new CUBLAS(worker)
    let rng = System.Random(42)
    use dmat1 = worker.Malloc(Array.init (m * n) (fun _ -> rng.NextDouble() |> float32))
    use dmat2 = worker.Malloc(Array.init (m * n) (fun _ -> rng.NextDouble() |> float32))
    use dmatr = worker.Malloc<float32>(m * n)
    worker.Eval <| fun _ ->
        let timer = System.Diagnostics.Stopwatch.StartNew()
        for i = 1 to iters do
            cublas.Sgeam(cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, m, n, 2.0f, dmat1.Ptr, m, 2.0f, dmat2.Ptr, m , dmatr.Ptr, m)
        if sync then worker.Synchronize()
        timer.Stop()
        printfn "%f ms / %d %s (cublas + worker.eval)" timer.Elapsed.TotalMilliseconds iters (if sync then "sync" else "nosync")

let test worker m n sync iters =
    test6 worker m n sync iters
    test7 worker m n sync iters
    test1 worker m n sync iters
    test2 worker m n sync iters
    test3 worker m n sync iters
    test4 worker m n sync iters
    test5 worker m n sync iters

let testReduce1 (worker:Worker) n iters =
    let rng = System.Random(42)
    use input = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use reduceModule = new DeviceReduceModule<float32>(GPUModuleTarget.Worker(worker), <@ (+) @>)
    // JIT compile and load GPU code for this module
    reduceModule.GPUForceLoad()
    // create a reducer which will allocate temp memory for maxNum=n
    let reduce = reduceModule.Create(n)
    let timer = System.Diagnostics.Stopwatch.StartNew()
    for i = 1 to 10000 do
        reduce.Reduce(input.Ptr, n) |> ignore
    timer.Stop()
    printfn "%f ms / %d (pre-load gpu code)" timer.Elapsed.TotalMilliseconds iters

let testReduce2 (worker:Worker) n iters =
    let rng = System.Random(42)
    use input = worker.Malloc(Array.init n (fun _ -> rng.NextDouble() |> float32))
    use reduceModule = new DeviceReduceModule<float32>(GPUModuleTarget.Worker(worker), <@ (+) @>)
    // JIT compile and load GPU code for this module
    reduceModule.GPUForceLoad()
    // create a reducer which will allocate temp memory for maxNum=n
    let reduce = reduceModule.Create(n)
    worker.Eval <| fun _ ->
        let timer = System.Diagnostics.Stopwatch.StartNew()
        for i = 1 to 10000 do
            reduce.Reduce(input.Ptr, n) |> ignore
        timer.Stop()
        printfn "%f ms / %d (pre-load gpu code and avoid thread switching)" timer.Elapsed.TotalMilliseconds iters

let testReduce worker n iters =
    testReduce1 worker n iters
    testReduce2 worker n iters

let workerDefault = Worker.Default
let worker = Worker.CreateOnCurrentThread(Device.Default)

test worker 1024 250 true 100000