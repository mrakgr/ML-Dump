// My first attempt at making reverse mode automatic differentiation.

// Everything here works well as long as there are no branches in the graph. I still have yet to implement fanout counters.
// Another problem is that the way I've structured the functions pretty much crushes the GC.


// In this version I've verified that the reason the program is crashing is because of the Alea's DeviceReduceModule.


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

    member this.Apply(n:int, x:deviceptr<float32>, y:deviceptr<float32>, z:deviceptr<float32>, coef_x, coef_y) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n coef_x x coef_y y z

    member this.Apply (coef_x, x: dMatrix, coef_y, y: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
        let z = this.GPUWorker.Malloc(x.dArray.Length)
        this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr, coef_x, coef_y)
        new dMatrix(x.num_rows, x.num_cols, z)

    member this.Apply (coef_x, x: dMatrix, coef_y, y: dMatrix, z: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceBinaryTransformModule"
        this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, coef_x, coef_y)
        z

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
    mutable A : dMatrix
    } with

    static member create (P: dMatrix) =
        {P=P;c=0;A=setModule.A(0.0f,P)}

type Rf =
    | DfR_Df_DM of Df_rec * (float32 -> dMatrix -> float32 -> dMatrix) * RDM
    | DfR_Df_Df of Df_rec * (float32 -> float32 -> float32 -> float32) * Rf

    member t.r =
        match t with
        | DfR_Df_DM(x,_,_) -> x
        | DfR_Df_Df(x,_,_) -> x

    member t.DisposeAll() =
        match t with
        | DfR_Df_Df _ -> ()
        | DfR_Df_DM(_,_,b) ->
            b.DisposeAll()

and RDM = 
    | DM of DM_rec
    | DMRb of DM_rec * (dMatrix -> dMatrix -> dMatrix -> dMatrix) * (dMatrix -> dMatrix -> dMatrix -> dMatrix) * RDM * RDM // Outside node * left derivative function * right derivative func * prev left node * prev right node.
    | DMRu of DM_rec * (dMatrix -> dMatrix -> dMatrix -> dMatrix) * RDM

    member t.r =
        match t with
        | DM x -> x
        | DMRb(x,_,_,_,_) -> x
        | DMRu(x,_,_) -> x

    member t.DisposeAll() =
            match t with
            | DM x -> ()
            | DMRb(x,_,_,l,r) ->
                x.A.Dispose()
                x.P.Dispose()
                l.DisposeAll()
                r.DisposeAll()
            | DMRu(x,_,b) -> 
                x.A.Dispose()
                x.P.Dispose()
                b.DisposeAll()
               
    static member makeNode(hidden_size, input_size) =
        let p = new dMatrix(hidden_size,input_size)
        DM (DM_rec.create p)

    static member makeNode(hidden_size, input_size, input: float32[]) =
        let p = new dMatrix(hidden_size,input_size, worker.Malloc(input))
        DM (DM_rec.create p)

    static member makeUniformRandomNode(hidden_size,input_size) =
        let scale = (2.0f / sqrt(hidden_size |> float32))
        let p = createRandomUniformMatrix hidden_size input_size scale (-scale/2.0f)
        DM (DM_rec.create p)


let matmult (a: RDM) (b:RDM) =
    let mm va vb =
        let c = sgemm nT nT 1.0f va vb
        let fl p r error = sgemm nT T 1.0f error r // The derivative with respect to the left. So the above argument gets inserted from the right left. Usually error * input.
        let fr p l error = sgemm T nT 1.0f l error // The derivative with respect to the right. So the above argument gets inserted from the right side. Usually weights * error.
        DMRb(DM_rec.create c,fl,fr,a,b)
    let va = a.r.P
    let vb = b.r.P
    mm va vb

/// Addition with broadcasting.
let addb a b = // b is for bias and a is for preactivations.
    let addb va vb =
        let c = addBias va vb
        let fl p r error = error
        let fr p l error = calculateBias 1.0f error
        DMRb(DM_rec.create c,fl,fr,a,b)
    let va = a.r.P
    let vb = b.r.P
    addb va vb

let sigmoidModule = new DeviceUnaryTransformModule <@ fun x -> 1.0f/(1.0f+exp(-x)) @>
let sigmoidErrorModule = new DeviceBinaryTransformModule <@ fun x out -> x*(1.0f-x)*out @>
let sigmoid (a:RDM) =
    let s va =
        let c = sigmoidModule.A(va)
        let fb p b error = sigmoidErrorModule.A(p,error)
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    s va

let add alpha a beta b =
    let add va vb =
        let c = sgeam nT nT alpha va beta vb
        let fl p r error = if alpha <> 1.0f then sgeam nT nT alpha error 0.0f r else error
        let fr p l error = if beta <> 1.0f then sgeam nT nT 0.0f l beta error else error
        DMRb(DM_rec.create c,fl,fr,a,b)
    let va = a.r.P
    let vb = b.r.P
    add va vb

let squareModule = new DeviceUnaryTransformModule <@ fun x -> x*x @>
let squareErrorModule = new DeviceBinaryTransformModule <@ fun x out -> 2.0f*x*out @>
let square a =
    let s va =
        let c = squareModule.A(va)
        let fb p b error = squareErrorModule.A(b,error)
        DMRu(DM_rec.create c,fb,a)
    let va = a.r.P
    s va

let sumModule = new DeviceUnaryMapSumModule <@ fun x -> x @>
let sumErrorModule = setModule//new DeviceUnaryCoefTransformModule <@ fun error _ -> error @> // I made a mistake here thinking it was error*a. Really the derivative of the sum is just the error.
let sum a =
    let s (va: dMatrix) =
        let c = sumModule.A(va)
        let fb p b error = sumErrorModule.A(error,b)
        DfR_Df_DM(Df_rec.create c,fb,a)
    let va = a.r.P
    s va

//let scaleModule = new DeviceUnaryCoefTransformModule <@ fun c x -> c*x @>
//let scaleErrorModule = new DeviceBinaryCoefTransformModule <@ fun c x _ out -> c*x*out @>
//I forgot that the function should be for scalars.
let scale alpha a =
    let s va =
        let c = alpha*va
        let fb p b error = alpha*error
        DfR_Df_Df(Df_rec.create c,fb,a)
    let va = a.r.P
    s va

let rec propagateReverseRDM (error: RDM) =
    match error with
    | DM _ -> ()
    | DMRb(p,fl,fr,l,r) ->
        let cl = fl p.P r.r.P error.r.A
        let cr = fr p.P l.r.P error.r.A
        l.r.A <- cl
        r.r.A <- cr
        propagateReverseRDM l
        propagateReverseRDM r
    | DMRu(p,fb,b) ->
        let c = fb p.P b.r.P error.r.A
        b.r.A <- c
        propagateReverseRDM b

let rec propagateReverseRDf (error: Rf) =
    match error with
    | DfR_Df_Df(p,fb,b) ->
        b.r.A <- fb p.P b.r.P p.A // This should also decrease the counter by one and add to the adjoint instead of setting it, but nevermind that for now.
        propagateReverseRDf b
    | DfR_Df_DM(p,fb,b) ->
        b.r.A <- fb p.P b.r.P p.A // This should also decrease the counter by one and add to the adjoint instead of setting it, but nevermind that for now.
        propagateReverseRDM b

//let w1 = RDM.makeUniformRandomNode(3,2)
//let bias1 = RDM.makeUniformRandomNode(3,1)
//let w2 = RDM.makeUniformRandomNode(1,3)
//let bias2 = RDM.makeUniformRandomNode(1,1)

let w1 = RDM.makeNode(3,2,[|0.5f;0.4f;0.3f;0.2f;0.1f;0.0f|])
let bias1 = RDM.makeNode(3,1,[|0.5f;0.4f;0.3f|])
let w2 = RDM.makeNode(1,3,[|-0.55f;-0.4f;-0.25f|])
let bias2 = RDM.makeNode(1,1,[|-0.8f|])

let input = RDM.makeNode(2,4,[|0.0f;0.0f;0.0f;1.0f;1.0f;0.0f;1.0f;1.0f;|])
let output = RDM.makeNode(1,4,[|0.0f;1.0f;1.0f;0.0f|])

let nodes = [|w1;bias1;w2;bias2|]
let f_batch_size = 4.0f
let learning_rate = 100.0f / f_batch_size

#time
let test num_iters =
    for i=1 to num_iters do
        let z1 = addb (matmult w1 input) bias1
        let a1 = sigmoid z1

        let z2 = addb (matmult w2 a1) bias2
        let a2 = sigmoid z2

        let r = add 1.0f output -1.0f a2
        let r2 = square r
        let r3 = sum r2
        let r4 = scale (1.0f/(2.0f*f_batch_size)) r3

        printfn "The cost is %f at iteration %i" r4.r.P i

        r4.r.A <- 1.0f
        propagateReverseRDf r4

        //printfn "z1=%A" (z1.r.A.dArray.Gather())
        //printfn "b.A=%A" (bias1.r.A.dArray.Gather())

        // Add gradients.
        for x in nodes do
            sgeam2 nT nT 1.0f x.r.P -learning_rate x.r.A x.r.P

        r4.DisposeAll()

        //printfn "%A" (w2.r.A.dArray.Gather())
test 1000
GC.Collect()
test 1000
#time
