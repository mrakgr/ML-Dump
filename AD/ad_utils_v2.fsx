#I @"C:\F# Packages\packages\Alea.CUDA.2.2.0.3307\lib\net40"
#I @"C:\F# Packages\packages\Alea.CUDA.IL.2.2.0.3307\lib\net40"
#I @"C:\F# Packages\packages\Alea.CUDA.Unbound.2.2.0.3307\lib\net40"
#I @"C:\F# Packages\packages\Alea.IL.2.2.0.3307\lib\net40"
#I @"C:\F# Packages\packages\Alea.CUDA.2.2.0.3307\private"

#r @"Alea.CUDA.Unbound.dll"
#r @"Alea.CUDA.IL.dll"
#r @"Alea.IL.dll"
#r @"Alea.CUDA.dll"
#r "System.Configuration.dll"

#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\FSharp.Charting.0.90.13\lib\net40\FSharp.Charting.dll"
#r @"C:\Program Files (x86)\Reference Assemblies\Microsoft\Framework\.NETFramework\v4.6\System.Windows.Forms.DataVisualization.dll"

open FSharp.Charting

open System
open System.IO

open Alea.CUDA

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- @"C:\F# Packages\packages\Alea.CUDA.2.2.0.3307\private"
    
// Resource.Path can be any empty or non-existing director like "C:\Temp\Alea Assemblies"
Alea.CUDA.Settings.Instance.Resource.Path <- @"C:\F# Packages\packages\Alea Assemblies"

open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open Alea.CUDA.CULib.CUBLASInterop
open Alea.CUDA.CULib.CUDNNInterop
open Alea.CUDA.IL
open Alea.CUDA.Unbound.Rng
open Alea.CUDA.Unbound
open Microsoft.FSharp.Quotations

open System.Collections

let worker = Worker.CreateThreadWorker(Device.Default)

let cublas = new CUBLAS(worker)

///Not transpose.
let nT = cublasOperation_t.CUBLAS_OP_N
///Transpose.
let T = cublasOperation_t.CUBLAS_OP_T

type dMatrix(num_rows:int,num_cols,dArray: DeviceMemory<float32>) = 
    inherit DisposableObject()

    new(num_rows,num_cols) =
        new dMatrix(num_rows,num_cols,worker.Malloc<float32>(num_rows*num_cols))

    member t.num_rows = num_rows
    member t.num_cols = num_cols
    member t.dArray = dArray

    override net.Dispose(disposing:bool) =
        if disposing then
            dArray.Dispose()

/// General matrix-matrix multiply from cuBLAS.
let sgemm transa transb (alpha: float32) (A:dMatrix) (B:dMatrix) =
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" a_col b_row)
    let m = if transa = nT then A.num_rows else A.num_cols
    let n = if transb = nT then B.num_cols else B.num_rows
    let k = a_col

    let lda = if transa = nT then m else k
    let ldb = if transb = nT then k else n
    let ldc = m

    let C_dArray = worker.Malloc<float32>(m*n)
    cublas.Sgemm(transa, transb, m, n, k, alpha, A.dArray.Ptr, lda, B.dArray.Ptr, ldb, 0.0f, C_dArray.Ptr, ldc)
    new dMatrix(m,n,C_dArray)

/// General matrix-matrix addition.
let sgeam transa transb (alpha: float32) (A:dMatrix) beta (B:dMatrix) =
    let a_row = if transa = nT then A.num_rows else A.num_cols
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    let b_col = if transb = nT then B.num_cols else B.num_rows
        
    if a_row <> b_row then failwith (sprintf "a_row <> b_row in sgeam! %i <> %i" a_row b_row)
    if a_col <> b_col then failwith (sprintf "a_col <> b_col in sgeam! %i <> %i" a_col b_col)

    let lda = if transa = nT then a_row else a_col
    let ldb = if transa = nT then b_row else b_col
    let ldc = a_row

    let C_dArray = worker.Malloc<float32>(A.num_cols*A.num_rows)
    if A.dArray.Length <> B.dArray.Length then failwith "A.dArray.Length <> B.dArray.Length in sgeam"
    cublas.Sgeam(transa, transb, a_row, a_col, alpha, A.dArray.Ptr, lda, beta, B.dArray.Ptr, ldb, C_dArray.Ptr, ldc)
    new dMatrix(a_row,a_col,C_dArray)

/// General matrix-matrix addition.
let sgeam2 transa transb (alpha: float32) (A:dMatrix) beta (B:dMatrix) (C:dMatrix) =
    let a_row = if transa = nT then A.num_rows else A.num_cols
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    let b_col = if transb = nT then B.num_cols else B.num_rows
        
    if a_row <> b_row then failwith (sprintf "a_row <> b_row in sgeam2! %i <> %i" a_row b_row)
    if a_col <> b_col then failwith (sprintf "a_col <> b_col in sgeam2! %i <> %i" a_col b_col)

    if a_row <> C.num_rows then failwith (sprintf "a_row <> C.num_rows in sgeam2! %i <> %i" a_col b_col)
    if a_col <> C.num_cols then failwith (sprintf "a_col <> C.num_cols in sgeam2! %i <> %i" a_col b_col)

    let lda = if transa = nT then a_row else a_col
    let ldb = if transa = nT then b_row else b_col
    let ldc = a_row

    if A.dArray.Length <> B.dArray.Length then failwith "A.dArray.Length <> B.dArray.Length in sgeam2"
    if A.dArray.Length <> C.dArray.Length then failwith "A.dArray.Length <> C.dArray.Length in sgeam2"
    cublas.Sgeam(transa, transb, a_row, a_col, alpha, A.dArray.Ptr, lda, beta, B.dArray.Ptr, ldb, C.dArray.Ptr, ldc)
    //worker.Synchronize()


let inline copy_matrix A = sgeam nT nT 1.0f A 0.0f A

type DeviceUnaryMapSumModule(target, op:Expr<float32 -> float32 >) =
    inherit GPUModule(target)

    let block_size = 128
    let blockReducer = BlockReduce.RakingCommutativeOnly<float32>(dim3(block_size,1,1),worker.Device.Arch)

    new (op:Expr<float32 -> float32 >) =
        new DeviceUnaryMapSumModule(GPUModuleTarget.Worker(worker), op)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (x:deviceptr<float32>) (z: deviceptr<float32>) =
        let temp_storage = blockReducer.TempStorage.AllocateShared()
        let start = blockIdx.x * blockDim.x + threadIdx.x

        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        let mutable acc = __default_value<float32>()
        while i < n do
            acc <- acc + (__eval(op) x.[i])
            i <- i + stride
        let out_partial = blockReducer.Reduce(temp_storage, acc, fun a b -> a+b)
        if threadIdx.x = 0 then (__atomic_add z out_partial) |> ignore
            
    member this.A(n:int, x:deviceptr<float32>) =
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        let lp = LaunchParam(gridSize, block_size)
        use z = worker.Malloc([|0.0f|])
        this.GPULaunch <@ this.Kernel @> lp n x z.Ptr
        z.GatherScalar()

    member this.A(x: dMatrix) =
        this.A(x.dArray.Length, x.dArray.Ptr)


/// Unary transform module for applying single functions to an array.
type DeviceUnaryTransformModule(target, op:Expr<float32 -> float32>) =
    inherit ILGPUModule(target)

    new (op:Expr<float32 -> float32>) =
        new DeviceUnaryTransformModule(GPUModuleTarget.Worker(worker), op)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (x:deviceptr<float32>) (y:deviceptr<float32>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            y.[i] <- __eval(op) x.[i] 
            i <- i + stride

    member this.A(n:int, x:deviceptr<float32>, y:deviceptr<float32>) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n x y

    member this.A(x: dMatrix) =
        let y = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.Ptr)
        new dMatrix(x.num_rows, x.num_cols, y)

type DeviceBinaryCoefTransformModule(target, op:Expr<float32 -> float32 -> float32 -> float32 -> float32>) =
    inherit GPUModule(target)

    new (op:Expr<float32 -> float32 -> float32 -> float32 -> float32>) =
        new DeviceBinaryCoefTransformModule(GPUModuleTarget.Worker(worker), op)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) coef_x (x:deviceptr<float32>) coef_y (y:deviceptr<float32>) (z:deviceptr<float32>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            z.[i] <- __eval(op) coef_x x.[i] coef_y y.[i]
            i <- i + stride

    member this.A(n:int, x:deviceptr<float32>, y:deviceptr<float32>, z:deviceptr<float32>, coef_x, coef_y) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n coef_x x coef_y y z

    member this.A (coef_x, x: dMatrix, coef_y, y: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
        let z = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr, coef_x, coef_y)
        new dMatrix(x.num_rows, x.num_cols, z)

/// Binary transform module for applying functions to two identically sized arrays.
/// Can be in-place or pointed to a destination.
type DeviceBinaryTransformModule(target, op:Expr<float32 -> float32 -> float32>) =
    inherit GPUModule(target)

    new (op:Expr<float32 -> float32 -> float32>) =
        new DeviceBinaryTransformModule(GPUModuleTarget.Worker(worker), op)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (z:deviceptr<float32>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            z.[i] <- __eval(op) x.[i] y.[i]
            i <- i + stride

    member this.A(n:int, x:deviceptr<float32>, y:deviceptr<float32>, z:deviceptr<float32>) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n x y z

    member this.A (x: dMatrix, y: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
        let z = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr)
        new dMatrix(x.num_rows, x.num_cols, z)

type DeviceUnaryCoefTransformModule(target, op:Expr<float32 -> float32 -> float32>) =
    inherit ILGPUModule(target)

    new (op:Expr<float32 -> float32 -> float32>) =
        new DeviceUnaryCoefTransformModule(GPUModuleTarget.Worker(worker), op)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) coef_x (x:deviceptr<float32>) (y:deviceptr<float32>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            y.[i] <- __eval(op) coef_x x.[i] 
            i <- i + stride

    member this.A(n:int, x:deviceptr<float32>, y:deviceptr<float32>, coef_x) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n coef_x x y 

    member this.A (coef_x, x: dMatrix) =
        let y = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.Ptr, coef_x)
        new dMatrix(x.num_rows, x.num_cols, y)

    member this.A (coef_x, x: dMatrix, y: dMatrix) =
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceUnaryTransformModule"
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, coef_x)
        //worker.Synchronize()

// Gradient clipping module.
let gradclipModule = 
    new DeviceUnaryCoefTransformModule
        <@ fun coef_a a ->
        if a > coef_a then coef_a
        else if a < -coef_a then -coef_a
        else a @>

let setModule = new DeviceUnaryCoefTransformModule <@ fun coef_x _ -> coef_x @>

let cudnn = new CUDNN(worker)
let biasTensorDesc = new CUDNNTensorDescriptor()
let dstTensorDesc = new CUDNNTensorDescriptor()

/// Adds the biases to the preactivations. Differs from the function in utils.fsx, in that it makes a copy of the preactivations and then broadcast adds to
/// that copy. I think this is redundant, but I want to take no chances during this run. The goal is to maximize correctness.
let addBias (preactivations: dMatrix) (bias: dMatrix) =
    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;
    biasTensorDesc.Set4D(TensorFormat, DataType, 1, 1, bias.num_rows, bias.num_cols)
    dstTensorDesc.Set4D(TensorFormat, DataType, 1, preactivations.num_cols, preactivations.num_rows, 1)
    let alpha, beta = 1.f, 1.f
    let copy_preact = copy_matrix preactivations
    cudnn.AddTensor(CUDNNInterop.cudnnAddMode_t.CUDNN_ADD_IMAGE, alpha, biasTensorDesc, bias.dArray.Ptr, beta, dstTensorDesc, copy_preact.dArray.Ptr)
    new dMatrix(copy_preact.num_rows,copy_preact.num_cols,copy_preact.dArray)

/// The reverse of the addBias function. It makes a copy of the bias parameters before adding to it.
/// Used to construct to the bias gradient from the error matrix.
let calculateBias alpha (error: dMatrix) =
    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NHWC;
    dstTensorDesc.Set4D(TensorFormat, DataType, 1, error.num_rows, 1, error.num_cols)
    
    let bias = new dMatrix(error.num_rows,1)
    biasTensorDesc.Set4D(TensorFormat, DataType, 1, bias.num_rows, 1, bias.num_cols)

    cudnn.ConvolutionBackwardBias(alpha,dstTensorDesc,error.dArray.Ptr,0.0f,biasTensorDesc,bias.dArray.Ptr)
    bias
    

let cudaRandomModule = new XorShift7.CUDA.DefaultUniformRandomModuleF32(GPUModuleTarget.Worker(worker))
let cudaRandom = cudaRandomModule.Create(50000,1,uint32 DateTime.Now.Millisecond) :> IRandom<float32>
    
let mutable stream_id = 0
/// This function has only two streams, so it can only create two non overlapping
/// arrays. Beware. For more arrays, increase the number of streams.
/// Current number of streams: 50000.
let createRandomUniformMatrix weights_num_rows weights_num_cols (scaling_factor : float32) location =
    let weights_total_size = weights_num_rows*weights_num_cols
        
    let cudaBuffer = cudaRandom.AllocCUDAStreamBuffer weights_total_size
    cudaRandom.Fill(stream_id,weights_total_size,cudaBuffer,scaling_factor,location)
    stream_id <- stream_id+1

    new dMatrix(weights_num_rows,weights_num_cols,cudaBuffer)

type Df_rec = {
    P: float32 
    mutable c : int 
    mutable A : float32
    } with

    static member create P =
        {P=P;c=0;A=0.0f}

type DM_rec = {
    P: dMatrix 
    mutable c : int 
    A : dMatrix
    } with

    static member create (P: dMatrix) =
        {P=P;c=0;A=setModule.A(0.0f,P)}

type Rf =
    | DfR_Df_DM of Df_rec * (float32 -> dMatrix) * RDM
    | DfR_Df_Df of Df_rec * (float32 -> float32) * Rf
    | DfR_Df_Df_Df of Df_rec * (float32 -> float32) * (float32 -> float32) * Rf * Rf

    member t.r =
        match t with
        | DfR_Df_DM(x,_,_) -> x
        | DfR_Df_Df(x,_,_) -> x
        | DfR_Df_Df_Df(x,_,_,_,_) -> x

and RDM = 
    | DM of DM_rec
    | DMRb of DM_rec * (dMatrix -> dMatrix) * (dMatrix -> dMatrix) * RDM * RDM // Outside node * left derivative function * right derivative func * prev left node * prev right node.
    | DMRu of DM_rec * (dMatrix -> dMatrix) * RDM

    member t.r =
        match t with
        | DM x -> x
        | DMRb(x,_,_,_,_) -> x
        | DMRu(x,_,_) -> x

    static member makeNode(hidden_size, input_size) =
        let p = new dMatrix(hidden_size,input_size)
        DM (DM_rec.create p)

    static member makeNode(hidden_size, input_size, input: float32[]) =
        let p = new dMatrix(hidden_size,input_size, worker.Malloc(input))
        DM (DM_rec.create p)

    static member makeUniformRandomNode(hidden_size,input_size) =
        let scale = (2.0f / sqrt(hidden_size+input_size |> float32))
        let p = createRandomUniformMatrix hidden_size input_size scale (-scale/2.0f)
        DM (DM_rec.create p)


// The type for the tape.
type R = 
    | Rf of Rf 
    | RDM of RDM
    
    member t.c =
        match t with
        | Rf x -> x.r.c
        | RDM x -> x.r.c

    member t.rDf =
        match t with
        | Rf x -> x.r
        | RDM x -> failwith "Invalid call to rDf"

    member t.rDM =
        match t with
        | Rf x -> x.r
        | RDM x -> failwith "Invalid call to rDf"

type tapeType = Generic.List<R>
let mutable tape = Generic.List<R>()

let matmult (a: RDM) (b:RDM) =
    let mm va vb =
        let c = sgemm nT nT 1.0f va vb
        let fl error = sgemm nT T 1.0f error vb // The derivative with respect to the left. So the above argument gets inserted from the right left. Usually error * input.
        let fr error = sgemm T nT 1.0f va error // The derivative with respect to the right. So the above argument gets inserted from the right side. Usually weights * error.
        DMRb(DM_rec.create c,fl,fr,a,b)
    let va = a.r.P
    let vb = b.r.P
    a.r.c <- a.r.c+1
    b.r.c <- b.r.c+1
    let t = mm va vb
    tape.Add(RDM t)
    t

/// Addition with broadcasting.
let addb a b = // b is for bias and a is for preactivations.
    let addb va vb =
        let c = addBias va vb
        let fl error = error
        let fr error = calculateBias 1.0f error
        DMRb(DM_rec.create c,fl,fr,a,b)
    let va = a.r.P
    let vb = b.r.P
    a.r.c <- a.r.c+1
    b.r.c <- b.r.c+1
    let t = addb va vb
    tape.Add(RDM t)
    t

let sigmoidModule = new DeviceUnaryTransformModule <@ fun x -> 1.0f/(1.0f+exp(-x)) @>
let sigmoidErrorModule = new DeviceBinaryTransformModule <@ fun x error -> x*(1.0f-x)*error @>
let sigmoid (a:RDM) =
    let s va =
        let c = sigmoidModule.A(va)
        let fb error = sigmoidErrorModule.A(c,error)
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = s va
    tape.Add(RDM t)
    t

let tanhModule = new DeviceUnaryTransformModule <@ fun x -> tanh x @>
let tanhErrorModule = new DeviceBinaryTransformModule <@ fun x error -> (1.0f-x*x)*error @>
let tanh_ (a:RDM) =
    let s va =
        let c = tanhModule.A(va)
        let fb error = tanhErrorModule.A(c,error)
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = s va
    tape.Add(RDM t)
    t

let add alpha a beta b =
    let add va vb =
        let c = sgeam nT nT alpha va beta vb
        let fl error = if alpha <> 1.0f then sgeam nT nT alpha error 0.0f error else error
        let fr error = if beta <> 1.0f then sgeam nT nT 0.0f error beta error else error
        DMRb(DM_rec.create c,fl,fr,a,b)
    let va = a.r.P
    let vb = b.r.P
    a.r.c <- a.r.c+1
    b.r.c <- b.r.c+1
    let t = add va vb
    tape.Add(RDM t)
    t

let squareModule = new DeviceUnaryTransformModule <@ fun x -> x*x @>
let squareErrorModule = new DeviceBinaryTransformModule <@ fun x out -> 2.0f*x*out @>
let square a =
    let s va =
        let c = squareModule.A(va)
        let fb error = squareErrorModule.A(va,error)
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = s va
    tape.Add(RDM t)
    t

let sumModule = new DeviceUnaryMapSumModule <@ fun x -> x @>
let sumErrorModule = setModule
let sum a =
    let s (va: dMatrix) =
        let c = sumModule.A(va)
        let fb error = sumErrorModule.A(error,va)
        DfR_Df_DM(Df_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = s va
    tape.Add(Rf t)
    t

let scale alpha a =
    let s va =
        let c = alpha*va
        let fb error = alpha*error
        DfR_Df_Df(Df_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = s va
    tape.Add(Rf t)
    t

let logModule = new DeviceUnaryTransformModule <@ fun x -> log x @>
let logErrorModule = new DeviceBinaryTransformModule <@ fun x error -> error / x  @>
let log_ a =
    let l va =
        let c = logModule.A(va)
        let fb error = logErrorModule.A(va, error)
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = l va
    tape.Add(RDM t)
    t

let hadamaradMultiplicationModule = new DeviceBinaryTransformModule <@ fun a b -> a*b @>
let hadamaradMultiplicationErrorModule = hadamaradMultiplicationModule//new DeviceBinaryTransformModule <@ fun a error -> a*error @>
let hadmult a b =
    let h va vb =
        let c = hadamaradMultiplicationModule.A(va,vb)
        let fl error = hadamaradMultiplicationErrorModule.A(vb,error)
        let fr error = hadamaradMultiplicationErrorModule.A(va,error)
        DMRb(DM_rec.create c,fl,fr,a,b)
    let va = a.r.P
    let vb = b.r.P
    a.r.c <- a.r.c+1
    b.r.c <- b.r.c+1
    let t = h va vb
    tape.Add(RDM t)
    t

let scalarAddModule = new DeviceUnaryCoefTransformModule <@ fun scalar x -> scalar + x @>
let scalar_add a b =
    let l va =
        let c = scalarAddModule.A(b,va)
        let fb error = error
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = l va
    tape.Add(RDM t)
    t

let scalarMatrixAddModule = new DeviceBinaryCoefTransformModule <@ fun scalar x coef _-> scalar + coef*x @>
let scalar_matrix_add scalar coef a =
    let l va =
        let c = scalarMatrixAddModule.A(scalar,va,coef,va)
        let fb error = if coef = 1.0f then error else sgeam nT nT coef error 0.0f error
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = l va
    tape.Add(RDM t)
    t

let neg a =
    let n va =
        let c = sgeam nT nT -1.0f va 0.0f va
        let fb error = sgeam nT nT -1.0f error 0.0f error
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    a.r.c <- a.r.c+1
    let t = n va
    tape.Add(RDM t)
    t

let linear_layer (mm: (RDM*RDM) []) (hh: (RDM*RDM) []) (bias: RDM option) =
    let mats = [|for l,r in mm do yield matmult l r|]
    let hads = [|for l,r in hh do yield hadmult l r|]
    let t = [|mats;hads|] |> Array.concat
    let sum = Array.fold (fun state x -> add 1.0f state 1.0f x) t.[0] t.[1..]
    match bias with
    | Some bias -> addb sum bias
    | None -> sum

let add_two_scalars (a: Rf) (b: Rf) =
    let s va vb =
        let c = va+vb
        let fb error = error
        DfR_Df_Df_Df(Df_rec.create c,fb,fb,a,b)
    let va = a.r.P
    let vb = b.r.P
    a.r.c <- a.r.c+1
    b.r.c <- b.r.c+1
    let t = s va vb
    tape.Add(Rf t)
    t

let sum_scalars (a: Rf[]) = Array.fold add_two_scalars a.[0] a.[1..]

let propagateReverseTape() =
    for i=tape.Count-1 downto 0 do
        let out = tape.[i]
        match out with
        | RDM out ->
            match out with
                | DM _ -> ()
                | DMRb(p,fl,fr,l,r) ->
                        sgeam2 nT nT 1.0f l.r.A 1.0f (fl p.A) l.r.A
                        sgeam2 nT nT 1.0f r.r.A 1.0f (fr p.A) r.r.A
                | DMRu(p,fb,b) ->
                        sgeam2 nT nT 1.0f b.r.A 1.0f (fb p.A) b.r.A
        | Rf out ->
            match out with
            | DfR_Df_Df(p,fb,b) ->
                    b.r.A <- b.r.A + (fb p.A)
            | DfR_Df_DM(p,fb,b) ->
                    sgeam2 nT nT 1.0f b.r.A 1.0f (fb p.A) b.r.A
            | DfR_Df_Df_Df(p,fl,fr,l,r) ->
                    l.r.A <- l.r.A + (fl p.A)
                    r.r.A <- r.r.A + (fr p.A)

let rec propagateReverseRDM (out: RDM) =
    match out with
    | DM _ -> false
    | DMRb(p,fl,fr,l,r) ->
        //printfn "I am in DMRb"
        //printfn "%A" p
        if p.c = 0 then
            p.c <- p.c-1
            l.r.c <- l.r.c-1
            sgeam2 nT nT 1.0f l.r.A 1.0f (fl p.A) l.r.A
            r.r.c <- r.r.c-1
            sgeam2 nT nT 1.0f r.r.A 1.0f (fr p.A) r.r.A
            let b1 = propagateReverseRDM l
            let b2 = propagateReverseRDM r
            b1 || b2
        else if p.c > 0 then true
        else 
            let b1 = propagateReverseRDM l
            let b2 = propagateReverseRDM r
            b1 || b2
    | DMRu(p,fb,b) ->
        //printfn "I am in DMRu"
        //printfn "%A" p
        if p.c = 0 then
            p.c <- p.c-1
            b.r.c <- b.r.c-1
            sgeam2 nT nT 1.0f b.r.A 1.0f (fb p.A) b.r.A
            propagateReverseRDM b
        else if p.c > 0 then true
        else propagateReverseRDM b

let rec propagateReverseRDf (out: Rf) =
    match out with
    | DfR_Df_Df(p,fb,b) ->
        //printfn "I am in DfR_Df_Df"
        //printfn "%A" p
        if p.c = 0 then
            p.c <- p.c-1
            b.r.c <- b.r.c-1
            b.r.A <- b.r.A + (fb p.A)
            propagateReverseRDf b
        else if p.c > 0 then true
        else propagateReverseRDf b
    | DfR_Df_DM(p,fb,b) ->
        //printfn "I am in DfR_Df_DM"
        //printfn "%A" p
        if p.c = 0 then
            p.c <- p.c-1
            b.r.c <- b.r.c-1
            sgeam2 nT nT 1.0f b.r.A 1.0f (fb p.A) b.r.A
            propagateReverseRDM b
        else if p.c > 0 then true
        else propagateReverseRDM b
    | DfR_Df_Df_Df(p,fl,fr,l,r) ->
        //printfn "I am in DfR_Df_Df_Df"
        //printfn "%A" p
        if p.c = 0 then
            p.c <- p.c-1
            l.r.c <- l.r.c-1
            l.r.A <- l.r.A + (fl p.A)
            r.r.c <- r.r.c-1
            r.r.A <- r.r.A + (fr p.A)
            let b1 = propagateReverseRDf l
            let b2 = propagateReverseRDf r
            b1 || b2
        else if p.c > 0 then true
        else 
            let b1 = propagateReverseRDf l
            let b2 = propagateReverseRDf r
            b1 || b2

let cross_entropy_cost target activations =
    let log_activations = log_ activations
    let neg_cross_ent_l = hadmult target log_activations

    let neg_target_plus_one = scalar_matrix_add 1.0f -1.0f target

    let neg_activations_plus_one = scalar_matrix_add 1.0f -1.0f activations
    let log_neg_activations_plus_one = log_ neg_activations_plus_one

    let neg_cross_ent_r = hadmult neg_target_plus_one log_neg_activations_plus_one
    let cross_ent = add 1.0f neg_cross_ent_l 1.0f neg_cross_ent_r

    let s = sum cross_ent
    scale (-1.0f/float32 target.r.P.num_cols) s

let squared_error_cost target activations =
    let r = add 1.0f target -1.0f activations
    let r2 = square r
    let r3 = sum r2
    scale (0.5f/float32 target.r.P.num_cols) r3

let rng = System.Random()

// A recurrent layer of neurons
type Layer =
    {
    W:RDM  // Input weight matrix
    U:RDM  // Recurrent weight matrix
    b:RDM  // Bias vector
    a:RDM->RDM
    } with     // Activation function
     
    member l.ToArray = 
        [|l.W;l.U;l.b|]

    static member fromArray (a : RDM[]) act =
        {
         W = a.[0]
         U = a.[1]
         b = a.[2]
         a = act
        }

    static member createRandomLayer hidden_size input_size act =
        {
        W = RDM.makeUniformRandomNode(hidden_size, input_size)
        U = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b = RDM.makeUniformRandomNode(hidden_size, 1)
        a = act
        } 

    // For the section with no previous hidden state.
    member l.runLayerNoH (x:RDM) =
        linear_layer [|l.W,x|] [||] (Some l.b) |> l.a
    
    // For the section with no input
    member l.runLayerNoI (y:RDM) =
        linear_layer [|l.U,y|] [||] (Some l.b) |> l.a

    // For the section with previous hidden state
    member l.runLayer (x:RDM) (y:RDM) =
        linear_layer [|l.W,x;l.U,y|] [||] (Some l.b) |> l.a


type GRULayer =
    {W_u:RDM  // Input weight matrix for the update gate
     U_u:RDM  // Recurrent weight matrix for the update gate
     b_u:RDM  // Bias vector for the update gate

     W_r:RDM  // Input weight matrix for the reset gate
     U_r:RDM  // Recurrent weight matrix for the reset gate
     b_r:RDM  // Bias vector for the reset gate

     W_n:RDM  // Input weight matrix for the potential new state
     U_n:RDM  // Recurrent weight matrix for the potential new state
     b_n:RDM  // Bias vector for the potential new state

     a : RDM -> RDM
     } with
    
    /// Returns all the weights in an array.
    member l.ToArray =
        [|l.W_u;l.U_u;l.b_u;l.W_r;l.U_r;l.b_r;l.W_n;l.U_n;l.b_n|]

    static member createRandomGRULayer hidden_size input_size act =
        {
        W_u = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_u = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_u = RDM.makeUniformRandomNode(hidden_size, 1)

        W_r = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_r = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_r = RDM.makeUniformRandomNode(hidden_size, 1)

        W_n = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_n = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_n = RDM.makeUniformRandomNode(hidden_size, 1)

        a = act
        }

    // For the section with no previous hidden state.
    member l.runLayerNoH (x:RDM) =
        let update_gate = linear_layer [|l.W_u,x|] [||] (Some l.b_u) |> sigmoid
        let potential_new_state = linear_layer [|l.W_n,x|] [||] (Some l.b_n) |> l.a
        let output_b = hadmult (scalar_matrix_add 1.0f -1.0f update_gate) potential_new_state
        output_b
    
    // For the section with no input
    member l.runLayerNoI (y:RDM) =
        let update_gate = linear_layer [|l.U_u,y|] [||] (Some l.b_u) |> sigmoid
        let reset_gate = linear_layer [|l.U_r,y|] [||] (Some l.b_r) |> sigmoid
        let potential_new_state = linear_layer [|l.U_n, (hadmult reset_gate y)|] [||] (Some l.b_n) |> l.a
        linear_layer [||] [|update_gate,y;(scalar_matrix_add 1.0f -1.0f update_gate),potential_new_state|] None

    // For the section with previous hidden state
    member l.runLayer (x:RDM) (y:RDM) =
        let update_gate = linear_layer [|l.W_u,x;l.U_u,y|] [||] (Some l.b_u) |> sigmoid
        let reset_gate = linear_layer [|l.W_r,x;l.U_r,y|] [||] (Some l.b_r) |> sigmoid
        let potential_new_state = linear_layer [|l.W_n,x;l.U_n, (hadmult reset_gate y)|] [||] (Some l.b_n) |> l.a
        linear_layer [||] [|update_gate,y;(scalar_matrix_add 1.0f -1.0f update_gate),potential_new_state|] None

type LSTMLayer =
    {W_z:RDM  // Input weight matrix for the block input
     U_z:RDM  // Recurrent weight matrix for the block input
     b_z:RDM  // Bias vector for the block input

     W_i:RDM  // Input weight matrix for the input gate
     U_i:RDM  // Recurrent weight matrix for the input gate
     b_i:RDM  // Bias vector for the input gate
     P_i:RDM  // Peephole weight matrix for the input gate

     W_f:RDM  // Input weight matrix for the forget gate
     U_f:RDM  // Recurrent weight matrix for the forget gate
     b_f:RDM  // Bias vector for the forget gate
     P_f:RDM  // Peephole weight matrix for the forget gate

     W_o:RDM  // Input weight matrix for the output gate
     U_o:RDM  // Recurrent weight matrix for the output gate
     b_o:RDM  // Bias vector for the output gate
     P_o:RDM  // Peephole weight matrix for the output gate

     block_input_a : RDM -> RDM
     block_output_a : RDM -> RDM
     } with
    
    /// Returns all the weights in an array.
    member l.ToArray = [|l.W_z;l.U_z;l.b_z;l.W_i;l.U_i;l.b_i;l.P_i;l.W_f;l.U_f;l.b_f;l.P_f;l.W_o;l.U_o;l.b_o;l.P_o|]
    static member fromArray (a: RDM[]) block_input_a block_output_a =
        {
         W_z = a.[0]
         U_z = a.[1]
         b_z = a.[2]

         W_i = a.[3]
         U_i = a.[4]
         b_i = a.[5]
         P_i = a.[6]

         W_f = a.[7]
         U_f = a.[8]
         b_f = a.[9]
         P_f = a.[10]

         W_o = a.[11]
         U_o = a.[12]
         b_o = a.[13]
         P_o = a.[14]

         block_input_a = block_input_a
         block_output_a = block_output_a
        }

    static member createRandomLSTMLayer hidden_size input_size block_input_a block_output_a =
        {
        W_z = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_z = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_z = RDM.makeUniformRandomNode(hidden_size, 1)

        W_i = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_i = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_i = RDM.makeUniformRandomNode(hidden_size, 1)
        P_i = RDM.makeUniformRandomNode(hidden_size, hidden_size)

        W_f = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_f = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_f = RDM.makeUniformRandomNode(hidden_size, 1)
        P_f = RDM.makeUniformRandomNode(hidden_size, hidden_size)

        W_o = RDM.makeUniformRandomNode(hidden_size, input_size)
        U_o = RDM.makeUniformRandomNode(hidden_size, hidden_size)
        b_o = RDM.makeUniformRandomNode(hidden_size, 1)
        P_o = RDM.makeUniformRandomNode(hidden_size, hidden_size)

        block_input_a = block_input_a
        block_output_a = block_output_a
        }

    member l.runLayer (x:RDM) (y:RDM) (c:RDM) =
        let block_input = linear_layer [|l.W_z,x;l.U_z,y|] [||] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer [|l.W_i,x;l.U_i,y;l.P_i,c|] [||] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer [|l.W_f,x;l.U_f,y;l.P_f,c|] [||] (Some l.b_f) |> sigmoid
        let c' = linear_layer [||] [|block_input,input_gate;c,forget_gate|] None
        let output_gate = linear_layer [|l.W_o,x;l.U_o,y;l.P_o,c'|] [||] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

    member l.runLayerNoH (x:RDM) =
        let block_input = linear_layer [|l.W_z,x|] [||] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer [|l.W_i,x|] [||] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer [|l.W_f,x|] [||] (Some l.b_f) |> sigmoid
        let c' = hadmult block_input input_gate
        let output_gate = linear_layer [|l.W_o,x;l.P_o,c'|] [||] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

    member l.runLayerNoI (y:RDM) (c:RDM) =
        let block_input = linear_layer [|l.U_z,y|] [||] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer [|l.U_i,y;l.P_i,c|] [||] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer [|l.U_f,y;l.P_f,c|] [||] (Some l.b_f) |> sigmoid
        let c' = linear_layer [||] [|block_input,input_gate;c,forget_gate|] None
        let output_gate = linear_layer [|l.U_o,y;l.P_o,c'|] [||] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

let save_data filename (ar: RDM []) =
    let stream_data = File.OpenWrite(filename)
    let writer_data = new BinaryWriter(stream_data)

    // Magic number
    writer_data.Write(929856)

    writer_data.Write(ar.Length)
    for x in ar do
        writer_data.Write(x.r.P.num_rows)
        writer_data.Write(x.r.P.num_cols)
        let t = x.r.P.dArray.Gather()
        for f in t do writer_data.Write(f)

    writer_data.Close()
    stream_data.Close()

let load_data file_name is_constant =
    let stream_data = File.OpenRead(file_name)
    let reader_data = new BinaryReader(stream_data)

    let m = reader_data.ReadInt32()
    if m <> 929856 then failwith "Wrong file type in load_weights"

    let l = reader_data.ReadInt32()
    let weights = [|
        for i=1 to l do
            let num_rows = reader_data.ReadInt32()
            let num_cols = reader_data.ReadInt32()
            let ar = [|for x=1 to num_rows*num_cols do yield reader_data.ReadSingle()|]
            yield RDM.makeNode(num_rows,num_cols,ar)
        |]

    reader_data.Close()
    stream_data.Close()
    weights