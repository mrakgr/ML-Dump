namespace Mnist

module Utils =

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
    open Microsoft.FSharp.Quotations

    //Alea.CUDA.Settings.Instance.JITCompile.Level <- "Profiling"
    Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- @"C:\F# Packages\packages\Alea.CUDA.2.1.2.3274\private"

    let worker = Worker.CreateOnCurrentThread(Device.Default)

    let cublas = new CUBLAS(worker)

    type dMatrix<'T> = {
        num_rows : int
        num_cols : int
        dArray : DeviceMemory<'T>
        }

    type dM = dMatrix<float32>

    type d4DMatrix<'T> = {
        num_feature_maps : int
        num_channels : int
        num_rows : int
        num_cols : int
        dArray : DeviceMemory<'T>
        }

    type d4M = d4DMatrix<float32>

    (*
    let cudnn = new CUDNN(worker)
    let biasTensorDesc = new CUDNNTensorDescriptor()
    let dstTensorDesc = new CUDNNTensorDescriptor()

    /// Adds the biases to the preactivations.
    let addBias (preactivations: dM) (bias: dM) =
        let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
        let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;
        biasTensorDesc.Set4D(TensorFormat, DataType, 1, 1, bias.num_rows, bias.num_cols)
        dstTensorDesc.Set4D(TensorFormat, DataType, 1, preactivations.num_cols, preactivations.num_rows, 1)
        let alpha, beta = 1.f, 1.f
        cudnn.AddTensor(CUDNNInterop.cudnnAddMode_t.CUDNN_ADD_IMAGE, alpha, biasTensorDesc, bias.dArray.Ptr, beta, dstTensorDesc, preactivations.dArray.Ptr)
    *)

    let cudaRandomModule = new XorShift7.CUDA.DefaultUniformRandomModuleF32(GPUModuleTarget.Worker(worker))
    let cudaRandom = cudaRandomModule.Create(50,1,42u) :> IRandom<float32>
    
    let mutable stream_id = 0
    /// This function has only two streams, so it can only create two non overlapping
    /// arrays. Beware. For more arrays, increase the number of streams.
    /// Current number of streams: 50.
    let createRandomUniformMatrix weights_num_rows weights_num_cols (scaling_factor : float32) =
        let weights_total_size = weights_num_rows*weights_num_cols
        
        let cudaBuffer = cudaRandom.AllocCUDAStreamBuffer weights_total_size
        cudaRandom.Fill(stream_id,weights_total_size,cudaBuffer,scaling_factor)
        stream_id <- stream_id+1

        {num_rows = weights_num_rows; num_cols = weights_num_cols; dArray = cudaBuffer} : dM

    let createRandomUniform4DMatrix weights_feature_maps weights_channels weights_num_rows weights_num_cols (scaling_factor : float32) =
        let weights_total_size = weights_feature_maps*weights_channels*weights_num_rows*weights_num_cols
        
        let cudaBuffer = cudaRandom.AllocCUDAStreamBuffer weights_total_size
        cudaRandom.Fill(stream_id,weights_total_size,cudaBuffer,scaling_factor)
        stream_id <- stream_id+1

        {num_feature_maps = weights_feature_maps; num_channels = weights_channels; num_rows = weights_num_rows; num_cols = weights_num_cols; dArray = cudaBuffer} : d4M

    /// Creates an uninitialized empty matrix.
    let createEmptyMatrix num_rows num_cols =
        {num_rows = num_rows; num_cols = num_cols; dArray = worker.Malloc<float32>(num_rows*num_cols)} : dM

    /// Create an empty matrix with the dimensions as the target.
    let createEmptyMatrixLike (target: dMatrix<'T>) =
        {num_rows = target.num_rows; num_cols = target.num_cols; dArray = worker.Malloc<'T>(target.num_rows*target.num_cols)} : dMatrix<'T>

    /// Creates an uninitialized empty matrix.
    let createEmptyMatrix4D num_feature_maps num_channels num_rows num_cols =
        {num_feature_maps=num_feature_maps; num_channels=num_channels; num_rows = num_rows; num_cols = num_cols; dArray = worker.Malloc<float32>(num_rows*num_cols)} : d4M

    /// Create an empty matrix with the dimensions as the target.
    let createEmptyMatrix4DLike (target: d4DMatrix<'T>) =
        {num_feature_maps=target.num_feature_maps; num_channels=target.num_channels; num_rows = target.num_rows; num_cols = target.num_cols; dArray = worker.Malloc<'T>(target.num_rows*target.num_cols)} : d4DMatrix<'T>

    ///Not transpose.
    let nT = cublasOperation_t.CUBLAS_OP_N
    ///Transpose.
    let T = cublasOperation_t.CUBLAS_OP_T

    /// General matrix-matrix multiply from cuBLAS with a destination parameter.
    let sgemm2 transa transb (alpha: float32) (A:dM) (B:dM) beta (C:dM) =
            let a_col = if transa = nT then A.num_cols else A.num_rows
            let b_row = if transb = nT then B.num_rows else B.num_cols
            if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm2! %i <> %i" a_col b_row)
            let m = if transa = nT then A.num_rows else A.num_cols
            let n = if transb = nT then B.num_cols else B.num_rows
            let k = a_col

            let lda = if transa = nT then m else k
            let ldb = if transb = nT then k else n
            let ldc = m

            if C.dArray.Length <> m*n then failwith "C.dArray.Length <> m*n in sgemm2"
            cublas.Sgemm(transa, transb, m, n, k, alpha, A.dArray.Ptr, lda, B.dArray.Ptr, ldb, beta, C.dArray.Ptr, ldc)
            {num_rows=m; num_cols=n; dArray=C.dArray}: dM

    /// General matrix-matrix multiply from cuBLAS.
    let sgemm transa transb (alpha: float32) (A:dM) (B:dM) =
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
            {num_rows=m; num_cols=n; dArray=C_dArray}: dM

    /// General matrix-matrix addition with an extra destination parameter.
    let sgeam2 transa transb (alpha: float32) (A:dM) beta (B:dM) (C:dM) =
            let a_row = if transa = nT then A.num_rows else A.num_cols
            let a_col = if transa = nT then A.num_cols else A.num_rows
            let b_row = if transb = nT then B.num_rows else B.num_cols
            let b_col = if transb = nT then B.num_cols else B.num_rows
        
            if a_row <> b_row then failwith (sprintf "a_row <> b_row in sgeam2! %i <> %i" a_row b_row)
            if a_col <> b_col then failwith (sprintf "a_col <> b_col in sgeam2! %i <> %i" a_col b_col)

            let lda = if transa = nT then a_row else a_col
            let ldb = if transa = nT then b_row else b_col
            let ldc = a_row

            if C.dArray.Length <> A.dArray.Length || A.dArray.Length <> B.dArray.Length then failwith "C.dArray.Length <> A.dArray.Length || A.dArray.Length <> B.dArray.Length in sgeam2"
            cublas.Sgeam(transa, transb, a_row, a_col, alpha, A.dArray.Ptr, lda, beta, B.dArray.Ptr, ldb, C.dArray.Ptr, ldc)
            {num_rows=a_row; num_cols=a_col; dArray=C.dArray}: dM

    /// General matrix-matrix addition.
    let inline sgeam transa transb (alpha: float32) (A:dM) beta (B:dM) =
            let C_dArray = worker.Malloc<float32>(A.num_cols*A.num_rows)
            let C = {num_rows=0; num_cols=0; dArray=C_dArray}: dM
            sgeam2 transa transb (alpha: float32) (A:dM) beta (B:dM) C

    let sgemv2 transa (alpha: float32) (A:dM) (B:dM) beta (C:dM) =
        let a_col = if transa = nT then A.num_cols else A.num_rows
        if a_col <> B.dArray.Length then failwith (sprintf "a_col <> B.dArray.Length in sgemv! %i <> %i" a_col B.dArray.Length)
        let m = A.num_rows
        let n = A.num_cols

        let lda = if transa = nT then m else n

        if C.dArray.Length <> lda then failwith "C.dArray.Length <> lda in sgemv2"
        cublas.Sgemv(transa,m,n,alpha,A.dArray.Ptr,lda,B.dArray.Ptr,1,beta,C.dArray.Ptr,1)
        {num_rows=lda; num_cols=1; dArray=C.dArray}: dM

    let sgemv transa (alpha: float32) (A:dM) (B:dM) =
        let a_col = if transa = nT then A.num_cols else A.num_rows
        if a_col <> B.dArray.Length then failwith (sprintf "n <> B.dArray.Length in sgemv! %i <> %i" a_col B.dArray.Length)
        let m = A.num_rows
        let n = A.num_cols

        let lda = if transa = nT then m else n

        let C = worker.Malloc<float32>(lda)
        cublas.Sgemv(transa,m,n,alpha,A.dArray.Ptr,lda,B.dArray.Ptr,1,0.0f,C.Ptr,1)
        {num_rows=lda; num_cols=1; dArray=C}: dM

    
    /// An extremely fast activation function. This one trully deserves the name
    /// of k-selection. Because row_size is determined statically and is a multiple of 32, 
    /// I've been able to unroll all the loops and store the variables into registers.

    /// 1.19 seconds per 10k iterations. Is only 3x slower than map and roughly 50x faster
    /// than sort.

    /// On Maxwell cards having the small block size of 32 is very efficient.
    /// http://arxiv.org/abs/1312.5663
    type sparsePiecewiseLinearActivationModule(target, num_rows, num_splits) =
        inherit GPUModule(target)

        let _ = if num_rows % 32 <> 0 then failwith "num_rows has to be a multiple of 32 in sparsePiecewiseLinearActivationModule"

        let grid_size = 384
        let block_size = 32

        new (num_rows, num_splits) = 
            new sparsePiecewiseLinearActivationModule(GPUModuleTarget.Worker(worker), num_rows, num_splits)
        /// Default number of splits=30
        new num_rows = new sparsePiecewiseLinearActivationModule(GPUModuleTarget.Worker(worker), num_rows, 30)

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

        member this.Apply((dmat: dM), k, (output: dM)) =
            if dmat.num_rows <> num_rows then failwith "dmat.num_rows <> num_rows sparsePiecewiseLinearActivationModule"
            if dmat.dArray.Length <> output.dArray.Length then failwith "dmat.dArray.Length <> output.dArray.Length in sparsePiecewiseLinearActivationModule"
            let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
            this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr output.dArray.Ptr k
            output

        member this.Apply((dmat: dM), k) =
            if dmat.num_rows <> num_rows then failwith "dmat.num_rows <> num_rows sparsePiecewiseLinearActivationModule"
            let output = createEmptyMatrixLike dmat
            let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
            this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr output.dArray.Ptr k
            output


    type DeviceBinaryMapReduceModule(target, op:Expr<float32 -> float32 -> float32>) =
        inherit GPUModule(target)

        let block_size = 128
        let blockReducer = BlockReduce.RakingCommutativeOnly<float32>(dim3(block_size,1,1),worker.Device.Arch)

        new (op:Expr<float32 -> float32 -> float32>) =
            new DeviceBinaryMapReduceModule(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (z: deviceptr<float32>) =
            let temp_storage = blockReducer.TempStorage.AllocateShared()
            let start = blockIdx.x * blockDim.x + threadIdx.x

            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            let mutable acc = __default_value<float32>()
            while i < n do
                acc <- acc + (__eval(op) x.[i] y.[i])
                i <- i + stride
            let out_partial = blockReducer.Reduce(temp_storage, acc, fun a b -> a+b)
            if threadIdx.x = 0 then (__atomic_add z out_partial) |> ignore
            
        member this.Apply(n:int, x:deviceptr<float32>, y:deviceptr<float32>) =
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
            let lp = LaunchParam(gridSize, block_size)
            let z = worker.Malloc([|0.0f|])
            this.GPULaunch <@ this.Kernel @> lp n x y z.Ptr
            z.Gather().[0]

        member this.Apply (x: dMatrix<float32>, y: dMatrix<float32>) =
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryMapReduceModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr)

    type DeviceUnaryMapReduceModule(target, op:Expr<float32 -> float32 >) =
        inherit GPUModule(target)

        let block_size = 128
        let blockReducer = BlockReduce.RakingCommutativeOnly<float32>(dim3(block_size,1,1),worker.Device.Arch)

        new (op:Expr<float32 -> float32 >) =
            new DeviceUnaryMapReduceModule(GPUModuleTarget.Worker(worker), op)

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
            
        member this.Apply(n:int, x:deviceptr<float32>) =
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
            let lp = LaunchParam(gridSize, block_size)
            let z = worker.Malloc([|0.0f|])
            this.GPULaunch <@ this.Kernel @> lp n x z.Ptr
            z.Gather().[0]

        member this.Apply (x: dMatrix<float32>) =
            this.Apply(x.dArray.Length, x.dArray.Ptr)

    (*
    type DeviceTrinaryMapReduceModule(target, op:Expr<float32 -> float32 -> float32 -> float32>) =
        inherit GPUModule(target)

        let block_size = 128
        let blockReducer = BlockReduce.RakingCommutativeOnly<float32>(dim3(block_size,1,1),worker.Device.Arch)

        new (op:Expr<float32 -> float32 -> float32 -> float32>) =
            new DeviceTrinaryMapReduceModule(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (w:deviceptr<float32>) (z: deviceptr<float32>) =
            let temp_storage = blockReducer.TempStorage.AllocateShared()
            let start = blockIdx.x * blockDim.x + threadIdx.x

            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            let mutable acc = __default_value<float32>()
            while i < n do
                acc <- acc + (__eval(op) x.[i] y.[i] w.[i])
                i <- i + stride
            let out_partial = blockReducer.Reduce(temp_storage, acc, fun a b -> a+b)
            if threadIdx.x = 0 then (__atomic_add z out_partial) |> ignore
            
        member this.Apply(n:int, x:deviceptr<float32>, y:deviceptr<float32>, s:deviceptr<float32>) =
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
            let lp = LaunchParam(gridSize, block_size)
            let z = worker.Malloc([|0.0f|])
            this.GPULaunch <@ this.Kernel @> lp n x y s z.Ptr
            z.Gather().[0]

        member this.Apply (x: dMatrix<float32>, y: dMatrix<float32>, s: dMatrix<float32>) =
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryMapReduceModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, s.dArray.Ptr)

    *)
    /// Trinary transform module for applying maping functions to three identically sized arrays.
    /// Can be in-place or pointed to a destination.
    type DeviceTrinaryTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T -> 'T>) =
        inherit GPUModule(target)

        new (op:Expr<'T -> 'T -> 'T -> 'T>) =
            new DeviceTrinaryTransformModule<'T>(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (z:deviceptr<'T>) (s:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                s.[i] <- __eval(op) x.[i] y.[i] z.[i]
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>, s:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y z s

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>) =
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceTrinaryTransformModule"
            let s = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = s}

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>, s: dMatrix<'T>) =
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length in DeviceTrinaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.dArray.Ptr)
            s

    /// Binary transform module for applying functions to two identically sized arrays.
    /// Can be in-place or pointed to a destination.
    type DeviceBinaryTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T>) =
        inherit GPUModule(target)

        new (op:Expr<'T -> 'T -> 'T>) =
            new DeviceBinaryTransformModule<'T>(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (z:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                z.[i] <- __eval(op) x.[i] y.[i]
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y z

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>) =
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
            let z = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = z}

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>) =
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceBinaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr)
            z

    /// Unary transform module for applying single functions to an array.
    /// Can be in-place or pointed to a destination.
    type DeviceUnaryTransformModule<'T>(target, op:Expr<'T -> 'T>) =
        inherit ILGPUModule(target)

        new (op:Expr<'T -> 'T>) =
            new DeviceUnaryTransformModule<'T>(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                y.[i] <- __eval(op) x.[i] 
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y

        member this.Apply (x: dMatrix<'T>) =
            let y = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = y}

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>) =
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceUnaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr)
            y

    /// A module for doing reductions across rows.
    /// This one is specifically tailored to find the row index of an element's maximum.
    /// It is kind of crappy.
    type maxRowReduceModule<'T when 'T : comparison>(target) =
        inherit ILGPUModule(target)

        new () = new maxRowReduceModule<'T>(GPUModuleTarget.Worker(worker))

        [<Kernel;ReflectedDefinition>]
        member this.Kernel cols (x:deviceptr<'T>) (y:deviceptr<int>) rows =
            let tid = blockIdx.x * blockDim.x + threadIdx.x
            let start = tid*rows
            let mutable cost, row_index = x.[start], 0
            if tid < cols then
                for i=1 to rows-1 do
                    let new_cost = x.[start+i]
                    if cost < new_cost then
                        cost <- new_cost
                        row_index <- i
                y.[tid] <- row_index

        member this.Apply (x: dMatrix<'T>) =
            let blockSize = 256
            let gridSize = divup x.num_cols blockSize
            let lp = LaunchParam(gridSize, blockSize)
            let dy = this.GPUWorker.Malloc<int>(x.num_cols)
            this.GPULaunch <@ this.Kernel @> lp x.num_cols x.dArray.Ptr dy.Ptr x.num_rows
            dy

    open System.Drawing

    /// Adapted from Mnist dataset visualization for logistic regression.
    let make_bitmap_from_imageset (imageset : dM) row_size col_size num_rows num_cols =
        let map_slice_to_bitmap (slice : float32 []) (bitmap : Bitmap) start_x end_x start_y end_y =
            let mutable slice_ind = 0
            for x=start_x to end_x do
                for y=start_y to end_y do
                    let c = int (slice.[slice_ind])
                    slice_ind <- slice_ind+1
                    let color = Color.FromArgb(c,c,c)
                    bitmap.SetPixel(y,x,color) 
        let float_array = imageset.dArray.Gather()
        let format = System.Drawing.Imaging.PixelFormat.Format24bppRgb
        let bitmap_digit = new Bitmap(col_size*num_cols,row_size*num_rows,format)
        let mutable digits = 0
        for x=0 to num_rows-1 do
            for y=0 to num_cols-1 do
                let start_slice = digits*imageset.num_rows
                let end_slice = (digits+1)*imageset.num_rows-1
                let slice = float_array.[start_slice..end_slice]
                digits <- digits+1

                // Normalization steps for each column.
                let norm = sqrt(slice |> Array.fold (fun state x -> state + x*x) 0.0f)
                let normed_slice = slice |> Array.map ( fun x -> (x / norm) * 127.0f + 127.0f)

                let start_x = x*row_size
                let end_x = start_x+row_size-1
                let start_y = y*col_size
                let end_y = start_y+col_size-1

                if (end_x-start_x+1)*(end_y-start_y+1) <> imageset.num_rows then failwith "(end_x-start_x+1)*(end_y-start_y+1) <> imageset.num_rows"

                map_slice_to_bitmap normed_slice bitmap_digit start_x end_x start_y end_y
        bitmap_digit

    /// Fills the array with ones.
    let onesModule = new DeviceUnaryTransformModule<float32> <@ fun _ -> 1.0f @>

    let save_weights file (weights: dM) =
        let host_weights = weights.dArray.Gather()
        let stream_data = File.OpenWrite(file)
        let writer_data = new BinaryWriter(stream_data)

        let host_weights = weights.dArray.Gather()
        for x in host_weights do
            writer_data.Write(x)
        writer_data.Close()
        stream_data.Close()

    let load_weights_mnist file num_rows =
        let stream_data = File.OpenRead(file)
        let reader_data = new BinaryReader(stream_data)

        let l = int stream_data.Length/4

        let ar = [|for i=1 to l do yield reader_data.ReadSingle()|]

        reader_data.Close()
        stream_data.Close()
        {num_rows=num_rows; num_cols=l/num_rows; dArray=worker.Malloc(ar)}