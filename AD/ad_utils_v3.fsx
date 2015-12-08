// No data races like I had suspected. Instead there was a bug in the tape. It is gone now.

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
open System.Collections

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

    new(num_rows: int,num_cols) =
        let t = worker.Malloc<float32>(num_rows*num_cols)
        ////worker.Synchronize()
        new dMatrix(num_rows,num_cols,t)

    member t.num_rows = num_rows
    member t.num_cols = num_cols
    member t.dArray = dArray

    override net.Dispose(disposing:bool) =
        if disposing then
            dArray.Dispose()
            ////worker.Synchronize()

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
    ////worker.Synchronize()
    new dMatrix(m,n,C_dArray)

/// General matrix-matrix multiply from cuBLAS. Inplace version
let sgemm2 transa transb (alpha: float32) (A:dMatrix) (B:dMatrix) beta (C:dMatrix) =
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" a_col b_row)
    let m = if transa = nT then A.num_rows else A.num_cols
    let n = if transb = nT then B.num_cols else B.num_rows
    let k = a_col

    let lda = if transa = nT then m else k
    let ldb = if transb = nT then k else n
    let ldc = m

    let C_dArray = C.dArray
    if C.dArray.Length <> m*n then failwith "C.dArray.Length <> m*n in sgemm2"
    if m <> C.num_rows || n <> C.num_cols then failwith "m <> C.num_rows || n <> C.num_cols in sgemm2"
    cublas.Sgemm(transa, transb, m, n, k, alpha, A.dArray.Ptr, lda, B.dArray.Ptr, ldb, beta, C_dArray.Ptr, ldc)
    ////worker.Synchronize()

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
    ////worker.Synchronize()
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
    ////worker.Synchronize()

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
        ////worker.Synchronize()
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
        ////worker.Synchronize()

    member this.A(x: dMatrix) =
        let y = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.Ptr)
        new dMatrix(x.num_rows, x.num_cols, y)

    member this.A(x: dMatrix, y: dMatrix) =
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceUnaryTransformModule"
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr)

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
        ////worker.Synchronize()

    member this.A (x: dMatrix, y: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
        let z = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr)
        new dMatrix(x.num_rows, x.num_cols, z)

    member this.A (x: dMatrix, y: dMatrix, z: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceBinaryTransformModule"
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr)

/// Trinary transform module for applying maping functions to three identically sized arrays.
/// Can be in-place or pointed to a destination.
type DeviceTrinaryTransformModule(target, op:Expr<float32 -> float32 -> float32 -> float32>) =
    inherit GPUModule(target)

    new (op:Expr<float32 -> float32 -> float32 -> float32>) =
        new DeviceTrinaryTransformModule(GPUModuleTarget.Worker(worker), op)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (n:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (z:deviceptr<float32>) (s:deviceptr<float32>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start 
        while i < n do
            s.[i] <- __eval(op) x.[i] y.[i] z.[i]
            i <- i + stride

    member this.A(n:int, x:deviceptr<float32>, y:deviceptr<float32>, z:deviceptr<float32>, s:deviceptr<float32>) =
        let blockSize = 256
        let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
        let gridSize = min (16 * numSm) (divup n blockSize)
        let lp = LaunchParam(gridSize, blockSize)
        this.GPULaunch <@ this.Kernel @> lp n x y z s
        ////worker.Synchronize()

    member this.A (x: dMatrix, y: dMatrix, z: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
        if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
            failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceTrinaryTransformModule"
        let s = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.Ptr)
        new dMatrix(x.num_rows, x.num_cols, s)

    member this.A (x: dMatrix, y: dMatrix, z: dMatrix, s: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
        if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
            failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length in DeviceTrinaryTransformModule"
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.dArray.Ptr)

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
        //worker.Synchronize()

    member this.A (coef_x, x: dMatrix) =
        let y = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.Ptr, coef_x)
        new dMatrix(x.num_rows, x.num_cols, y)

    member this.A (coef_x, x: dMatrix, y: dMatrix) =
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceUnaryTransformModule"
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, coef_x)

// Gradient clipping module.
let gradclipModule = 
    new DeviceUnaryCoefTransformModule
        <@ fun coef_a a ->
        if a > coef_a then coef_a
        else if a < -coef_a then -coef_a
        else a @>

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
        //worker.Synchronize()

    member this.A (coef_x, x: dMatrix, coef_y, y: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
        let z = this.GPUWorker.Malloc(x.dArray.Length)
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr, coef_x, coef_y)
        new dMatrix(x.num_rows, x.num_cols, z)

    member this.A (coef_x, x: dMatrix, coef_y, y: dMatrix, z: dMatrix) =
        if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
            failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
        if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceBinaryTransformModule"
        this.A(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, coef_x, coef_y)


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
    //worker.Synchronize()
    new dMatrix(copy_preact.num_rows,copy_preact.num_cols,copy_preact.dArray)

/// The inplace version.
let addBias2 beta (preactivations: dMatrix) alpha (bias: dMatrix) =
    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;
    biasTensorDesc.Set4D(TensorFormat, DataType, 1, 1, bias.num_rows, bias.num_cols)
    dstTensorDesc.Set4D(TensorFormat, DataType, 1, preactivations.num_cols, preactivations.num_rows, 1)
    cudnn.AddTensor(CUDNNInterop.cudnnAddMode_t.CUDNN_ADD_IMAGE, alpha, biasTensorDesc, bias.dArray.Ptr, beta, dstTensorDesc, preactivations.dArray.Ptr)
    //worker.Synchronize()

/// The reverse of the addBias function. It makes a copy of the bias parameters before adding to it.
/// Used to construct to the bias gradient from the error matrix.
let calculateBias alpha (error: dMatrix) =
    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NHWC;
    dstTensorDesc.Set4D(TensorFormat, DataType, 1, error.num_rows, 1, error.num_cols)
    
    let bias = new dMatrix(error.num_rows,1)
    biasTensorDesc.Set4D(TensorFormat, DataType, 1, bias.num_rows, 1, bias.num_cols)

    cudnn.ConvolutionBackwardBias(alpha,dstTensorDesc,error.dArray.Ptr,0.0f,biasTensorDesc,bias.dArray.Ptr)
    //worker.Synchronize()
    bias

let calculateBias2 alpha (error: dMatrix) (dest: dMatrix) =
    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NHWC;
    dstTensorDesc.Set4D(TensorFormat, DataType, 1, error.num_rows, 1, error.num_cols)
    biasTensorDesc.Set4D(TensorFormat, DataType, 1, dest.num_rows, 1, dest.num_cols)

    cudnn.ConvolutionBackwardBias(alpha,dstTensorDesc,error.dArray.Ptr,1.0f,biasTensorDesc,dest.dArray.Ptr)
    //worker.Synchronize()
    

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
    //worker.Synchronize()
    stream_id <- stream_id+1

    new dMatrix(weights_num_rows,weights_num_cols,cudaBuffer)

let fillRandomUniformMatrix (m: dMatrix) (scaling_factor : float32) location =
    let weights_total_size = m.num_rows*m.num_cols
        
    cudaRandom.Fill(stream_id,weights_total_size,m.dArray.Ptr,scaling_factor,location)
    //worker.Synchronize()
    stream_id <- stream_id+1

type Df_rec = {
    P: float32 ref
    A : float32 ref
    is_constant : bool
    } with

    static member create P =
        {P=P;A=ref 0.0f;is_constant=false}
    static member createConstant P =
        {P=P;A=ref 0.0f;is_constant=true}

type DM_rec = {
    P: dMatrix 
    A : dMatrix
    is_constant : bool
    } with

    static member create (P: dMatrix) =
        {P=P;A=setModule.A(0.0f,P);is_constant=false}
        
    static member createConstant P =
        {P=P;A=setModule.A(0.0f,P);is_constant=true}

type Rf =
    | DfR_Df_DM of Df_rec * (unit -> unit) * (unit -> unit) * RDM
    | DfR_Df_Df of Df_rec * (unit -> unit) * (unit -> unit) * Rf
    | DfR_Df_Dfseq of Df_rec * (unit -> unit) * (unit -> unit) * Rf []

    member t.r =
        match t with
        | DfR_Df_DM(x,_,_,_) -> x
        | DfR_Df_Df(x,_,_,_) -> x
        | DfR_Df_Dfseq(x,_,_,_) -> x

    member t.triggerForward() =
        match t with
        | DfR_Df_DM(x,ff,_,_) -> ff()
        | DfR_Df_Df(x,ff,_,_) -> ff()
        | DfR_Df_Dfseq(x,ff,_,_) -> ff()

    member t.triggerBackward() =
        match t with
        | DfR_Df_DM(x,_,fb,_) -> fb()
        | DfR_Df_Df(x,_,fb,_) -> fb()
        | DfR_Df_Dfseq(x,_,fb,_) -> fb()

and RDM = 
    | DM of DM_rec
    | DMRb of DM_rec * ff: (unit -> unit) * fl: (unit -> unit) * fr: (unit -> unit) * RDM * RDM // Outside node * left derivative function * right derivative func * prev left node * prev right node.
    | DMRu of DM_rec * ff: (unit -> unit) * fb: (unit -> unit) * RDM
    | DMRlin of DM_rec * ff: (unit -> unit) * db: (unit -> unit) * nodes: RDM[]
    // Looking at it now, I just realized that all these types (except DM) are redundant and could be replaced with DMRlin.

    member t.r =
        match t with
        | DM x -> x
        | DMRb(x,_,_,_,_,_) -> x
        | DMRu(x,_,_,_) -> x
        | DMRlin(x,_,_,_) -> x

    member t.triggerForward() =
        match t with
        | DM _ -> ()
        | DMRb(x,ff,_,_,_,_) -> ff()
        | DMRu(x,ff,_,_) -> ff()
        | DMRlin(x,ff,_,_) -> ff()

    member t.triggerBackward() =
        match t with
        | DM _ -> ()
        | DMRb(x,_,fl,fr,_,_) -> fl(); fr()
        | DMRu(x,_,fb,_) -> fb()
        | DMRlin(_,_,fb,_) -> fb()

    static member makeNode(hidden_size, input_size) =
        let p = new dMatrix(hidden_size,input_size)
        DM (DM_rec.create p)

    static member makeNode(hidden_size, input_size, input: float32[]) =
        if hidden_size*input_size <> input.Length then failwith "hidden_size*input_size <> input.Length in makeNode."
        let p = new dMatrix(hidden_size,input_size, worker.Malloc(input))
        DM (DM_rec.create p)

    static member makeConstantNode(hidden_size, input_size, input: float32[]) =
        if hidden_size*input_size <> input.Length then failwith "hidden_size*input_size <> input.Length in makeConstantNode."
        let p = new dMatrix(hidden_size,input_size, worker.Malloc(input))
        DM (DM_rec.createConstant p)

    static member makeUniformRandomNode(hidden_size,input_size) =
        let scale = (2.0f / sqrt(hidden_size+input_size |> float32))
        let p = createRandomUniformMatrix hidden_size input_size scale (-scale/2.0f)
        DM (DM_rec.create p)

// The type for the tape.
type R = 
    | Rf of Rf 
    | RDM of RDM
    
    member t.resetAdjoint() =
        match t with
        | Rf x -> x.r.A := 0.0f
        | RDM x -> setModule.A(0.0f,x.r.A,x.r.A)

    member t.resetPrimal() =
        match t with
        | Rf x -> x.r.P := 0.0f
        | RDM x -> setModule.A(0.0f,x.r.P,x.r.P)

    member t.triggerForward() =
        match t with
        | Rf x -> x.triggerForward()
        | RDM x -> x.triggerForward()

    member t.triggerBackward() =
        match t with
        | Rf x -> x.triggerBackward()
        | RDM x -> x.triggerBackward()

type tapeType = System.Collections.Generic.List<R>
let mutable tape = tapeType(1000)

let hadamaradMultiplicationModule = new DeviceBinaryTransformModule <@ fun a b -> a*b @>
let hadamaradMultiplicationErrorModule = new DeviceTrinaryTransformModule <@ fun a b c-> a*b+c @>
let hadmult (a: RDM) (b: RDM) =
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let c = hadamaradMultiplicationModule.A(va, vb)

    let node = DM_rec.create c
    let error = node.A

    let ff () = hadamaradMultiplicationModule.A(va, vb, c)
    let fl () = hadamaradMultiplicationErrorModule.A(vb,error,el,el)
    let fr () = hadamaradMultiplicationErrorModule.A(va,error,er,er)
    let t = DMRb(node,ff,fl,fr,a,b)
    tape.Add(RDM t)
    t


/// This is an optimization of the linear layer because the best optimization is to remove operations entirely.
/// Doing it standardly involves too many unnecessary allocations.
let linear_layer (mm: (RDM*RDM) []) (hads: (RDM*RDM) []) (bias: RDM option) =
    let c =
        if mm.Length > 0 then 
            let l,r = mm.[0]
            let c = new dMatrix(l.r.P.num_rows,r.r.P.num_cols)
            setModule.A(0.0f,c,c)
            c
        else if hads.Length > 0 then
            let l,r = hads.[0]
            setModule.A(0.0f,l.r.P)
        else failwith "Invalid input into linear_layer."

    for l,r in mm do sgemm2 nT nT 1.0f l.r.P r.r.P 1.0f c
    for l,r in hads do hadamaradMultiplicationErrorModule.A(l.r.P, r.r.P, c, c)
    match bias with
    | Some bias -> addBias2 1.0f c 1.0f bias.r.P
    | None -> ()

    let node = DM_rec.create c
    let error = node.A

    let ff() =
        setModule.A(0.0f,c,c)
        
        for l,r in mm do sgemm2 nT nT 1.0f l.r.P r.r.P 1.0f c
        for l,r in hads do hadamaradMultiplicationErrorModule.A(l.r.P, r.r.P, c, c)

        match bias with
        | Some bias -> addBias2 1.0f c 1.0f bias.r.P // There is a strange bug in my code currently, so I will move this here to enforce sameness with the above.
        | None -> ()
    let fb() =
        for l,r in mm do
            if l.r.is_constant = false then sgemm2 nT T 1.0f error r.r.P 1.0f l.r.A
            if r.r.is_constant = false then sgemm2 T nT 1.0f l.r.P error 1.0f r.r.A
        for l,r in hads do 
            hadamaradMultiplicationErrorModule.A(error, r.r.P, l.r.A, l.r.A)
            hadamaradMultiplicationErrorModule.A(l.r.P, error, r.r.A, r.r.A)
        match bias with
        | Some bias -> calculateBias2 1.0f error bias.r.A
        | None -> ()
    let ar =
        [|
        for l,r in mm do yield l; yield r
        for l,r in hads do yield l; yield r
        match bias with
        | Some bias -> yield bias 
        | None -> () |]
    let t = DMRlin(node,ff,fb,ar)
    tape.Add(RDM t)
    t

let matmult (a: RDM) (b:RDM) =
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let c = sgemm nT nT 1.0f va vb
        
    let node = DM_rec.create c
    let error = node.A
        
    let ff () = sgemm2 nT nT 1.0f va vb 0.0f c
    let fl () = if a.r.is_constant = false then sgemm2 nT T 1.0f error vb 1.0f el// The derivative with respect to the left. So the above argument gets inserted from the right left. Usually error * input.
    let fr () = if b.r.is_constant = false then sgemm2 T nT 1.0f va error 1.0f er// The derivative with respect to the right. So the above argument gets inserted from the right side. Usually weights * error.
    let t = DMRb(node,ff,fl,fr,a,b)
    tape.Add(RDM t)
    t

/// Addition with broadcasting.
let addb (a: RDM) (b: RDM) = // b is for bias and a is for preactivations.
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let c = addBias va vb

    let node = DM_rec.create c
    let error = node.A

    let ff () = 
        sgeam2 nT nT 1.0f va 0.0f c c
        addBias2 1.0f c 1.0f vb
    let fl () = sgeam2 nT nT 1.0f el 1.0f error el
    let fr () = calculateBias2 1.0f error er
    let t = DMRb(node,ff,fl,fr,a,b)
    tape.Add(RDM t)
    t

let sigmoidModule = new DeviceUnaryTransformModule <@ fun x -> 1.0f/(1.0f+exp(-x)) @>
let sigmoidErrorModule = new DeviceTrinaryTransformModule <@ fun x error below -> x*(1.0f-x)*error + below @>
let sigmoid (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let c = sigmoidModule.A(va)

    let node = DM_rec.create c
    let error = node.A

    let ff () = sigmoidModule.A(va,c)
    let fb () = sigmoidErrorModule.A(c,error,el,el)
    let t = DMRu(node,ff,fb,a)
    tape.Add(RDM t)
    t

let tanhModule = new DeviceUnaryTransformModule <@ fun x -> tanh x @>
let tanhErrorModule = new DeviceTrinaryTransformModule <@ fun x error below -> (1.0f-x*x)*error + below @>
let tanh_ (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let c = tanhModule.A(va)

    let node = DM_rec.create c
    let error = node.A

    let ff () = tanhModule.A(va,c)
    let fb () = tanhErrorModule.A(c,error,el,el)
    let t = DMRu(node,ff,fb,a)
    tape.Add(RDM t)
    t

let add alpha (a: RDM) beta (b: RDM) =
    let va = a.r.P
    let vb = b.r.P
    let el = a.r.A
    let er = b.r.A

    let c = sgeam nT nT alpha va beta vb

    let node = DM_rec.create c
    let error = node.A

    let ff () = sgeam2 nT nT alpha va beta vb c
    let fl () = sgeam2 nT nT alpha error 1.0f el el
    let fr () = sgeam2 nT nT 1.0f er beta error er
    let t = DMRb(node,ff,fl,fr,a,b)
    tape.Add(RDM t)
    t

let linear_layer_ (mm: (RDM*RDM) []) (hh: (RDM*RDM) []) (bias: RDM option) =
    let mats = [|for l,r in mm do yield matmult l r|]
    let hads = [|for l,r in hh do yield hadmult l r|]
    let t = [|mats;hads|] |> Array.concat
    let sum = Array.fold (fun state x -> add 1.0f state 1.0f x) t.[0] t.[1..]
    match bias with
    | Some bias -> addb sum bias
    | None -> sum

let squareModule = new DeviceUnaryTransformModule <@ fun x -> x*x @>
let squareErrorModule = new DeviceTrinaryTransformModule <@ fun x error below -> 2.0f*x*error + below @>
let square (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let c = squareModule.A(va)

    let node = DM_rec.create c
    let error = node.A

    let ff () = squareModule.A(va,c)
    let fb () = squareErrorModule.A(va,error,el,el)
    let t = DMRu(node,ff,fb,a)
    tape.Add(RDM t)
    t

let sumModule = new DeviceUnaryMapSumModule <@ fun x -> x @>
let sumErrorModule = new DeviceUnaryCoefTransformModule <@ fun error b -> error + b @> // I made a mistake here thinking it was error*a. Really the derivative of the sum is just the error.
let sum (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let c = sumModule.A(va)

    let node = Df_rec.create (ref c)
    
    let ff () = node.P := sumModule.A(va)
    let fb () = sumErrorModule.A(!node.A,el,el)
    let t = DfR_Df_DM(node,ff,fb,a)
    tape.Add(Rf t)
    t

let scale (alpha: float32) (a:Rf) =
    let c = alpha * !a.r.P

    let node = Df_rec.create (ref c)

    let ff () = node.P := alpha * !a.r.P
    let fb () = a.r.A := alpha * !node.A + !a.r.A
    let t = DfR_Df_Df(node,ff,fb,a)
    tape.Add(Rf t)
    t

let sum_scalars (a:Rf[]) =

    let c = ref 0.0f
    for l in a do c := !c + !l.r.P

    let node = Df_rec.create c

    let ff () = 
        let c = ref 0.0f
        for l in a do c := !c + !l.r.P
        node.P := !c
    let fb () = 
        for l in a do l.r.A := !node.A + !l.r.A
    let t = DfR_Df_Dfseq(node,ff,fb,a)
    tape.Add(Rf t)
    t

let logModule = new DeviceUnaryTransformModule <@ fun x -> log x @>
let logErrorModule = new DeviceTrinaryTransformModule <@ fun x error below -> error / x + below @>
let log_ (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let c = logModule.A(va)

    let node = DM_rec.create c
    let error = node.A

    let ff () = logModule.A(va,c)
    let fb () = logErrorModule.A(va,error,el, el)
    let t = DMRu(node,ff,fb,a)
    tape.Add(RDM t)
    t

let scalarMatrixAddModule = new DeviceBinaryCoefTransformModule <@ fun scalar x coef _-> scalar + coef*x @>
let scalar_matrix_add scalar coef (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let c = scalarMatrixAddModule.A(scalar,va,coef,va)

    let node = DM_rec.create c
    let error = node.A

    let ff () = scalarMatrixAddModule.A(scalar,va,coef,va,c)
    let fb () = sgeam2 nT nT coef error 1.0f el el
    let t = DMRu(node,ff,fb,a)
    tape.Add(RDM t)
    t

let scalarAddModule = new DeviceUnaryCoefTransformModule <@ fun scalar x -> scalar + x @>
let scalar_add (a:RDM) b =
    let va = a.r.P
    let el = a.r.A

    let c = scalarAddModule.A(b,va)

    let node = DM_rec.create c
    let error = node.A

    let ff () = scalarAddModule.A(b,va,c)
    let fb () = sgeam2 nT nT 1.0f error 1.0f el el
    let t = DMRu(node,ff,fb,a)
    tape.Add(RDM t)
    t

let neg (a:RDM) =
    let va = a.r.P
    let el = a.r.A

    let c = sgeam nT nT -1.0f va 0.0f va

    let node = DM_rec.create c
    let error = node.A

    let ff () = sgeam2 nT nT -1.0f va 0.0f va c
    let fb () = sgeam2 nT nT -1.0f error 1.0f el el
    let t = DMRu(node,ff,fb,a)
    tape.Add(RDM t)
    t

let cross_entropy_cost target activations =
    let cross_ent = linear_layer [||] [|target,log_ activations;scalar_matrix_add 1.0f -1.0f target, log_ (scalar_matrix_add 1.0f -1.0f activations)|] None
    let s = sum cross_ent
    scale (-1.0f/float32 target.r.P.num_cols) s

let squared_error_cost target activations =
    let r1 = add 1.0f target -1.0f activations
    let r2 = square r1
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

let forwardpropTape (tape: Generic.List<R>) = for i=0 to tape.Count-1 do tape.[i].triggerForward()
let reversepropTape (tape: Generic.List<R>) = for i=tape.Count-1 downto 0 do tape.[i].triggerBackward()
let resetTapeAdjoint (tape: Generic.List<R>) = for x in tape do x.resetAdjoint()
let resetTapePrimal (tape: Generic.List<R>) = for x in tape do x.resetPrimal()

let add_gradients_to_weights (base_nodes: RDM[]) learning_rate clip_coef = 
    for x in base_nodes do 
        gradclipModule.A(clip_coef,x.r.A,x.r.A)
        sgeam2 nT nT 1.0f x.r.P -learning_rate x.r.A x.r.P

let nesterov_add_gradients (base_nodes: RDM[]) (momentum_matrices: dMatrix[]) (copy_weights: dMatrix[]) learning_rate momentum_rate clip_coef = 
    for i=0 to base_nodes.Length-1 do
        let x = base_nodes.[i] 
        let m = momentum_matrices.[i]
        let c = copy_weights.[i]
        gradclipModule.A(clip_coef,x.r.A,x.r.A)
        sgeam2 nT nT -learning_rate x.r.A momentum_rate m m // Add the gradients to the momentum matrices
        sgeam2 nT nT 1.0f m 1.0f c c // Add momentum to the copy matrix
        sgeam2 nT nT 1.0f c momentum_rate m x.r.P // Apply Nesterov's momentum to the weights. It is really the copy weights that serve as the basis.

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
            match is_constant with
            | true -> yield RDM.makeConstantNode(num_rows,num_cols,ar)
            | false -> yield RDM.makeNode(num_rows,num_cols,ar)
        |]

    reader_data.Close()
    stream_data.Close()
    weights

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