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

module Utils =

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

    let calculateBias alpha (error: dM) beta (bias: dM) =
        let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
        let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;
        dstTensorDesc.Set4D(TensorFormat, DataType, 1, error.num_rows, 1, error.num_cols)
        biasTensorDesc.Set4D(TensorFormat, DataType, 1, bias.num_rows, 1, bias.num_cols)
        cudnn.ConvolutionBackwardBias(alpha/float32 error.num_cols,dstTensorDesc,error.dArray.Ptr,beta,biasTensorDesc,bias.dArray.Ptr)

    /// Sets beta (the momentum flag variable) to 1.0f after it is done.
    let dynamicCalculateBias alpha (error: dM) beta (bias: dM) =
        let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
        let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;
        dstTensorDesc.Set4D(TensorFormat, DataType, 1, error.num_rows, 1, error.num_cols)
        biasTensorDesc.Set4D(TensorFormat, DataType, 1, bias.num_rows, 1, bias.num_cols)
        cudnn.ConvolutionBackwardBias(alpha/float32 error.num_cols,dstTensorDesc,error.dArray.Ptr,!beta,biasTensorDesc,bias.dArray.Ptr)
        beta := 1.0f


    let cudaRandomModule = new XorShift7.CUDA.DefaultUniformRandomModuleF32(GPUModuleTarget.Worker(worker))
    let cudaRandom = cudaRandomModule.Create(50000,1,uint32 DateTime.Now.Millisecond) :> IRandom<float32>
    
    let mutable stream_id = 0
    /// This function has only two streams, so it can only create two non overlapping
    /// arrays. Beware. For more arrays, increase the number of streams.
    /// Current number of streams: 50000.
    let fillRandomUniformMatrix (scaling_factor : float32) (target: dM) =
        let weights_total_size = target.num_rows*target.num_cols
        
        cudaRandom.Fill(stream_id,weights_total_size,target.dArray.Ptr,scaling_factor)
        stream_id <- stream_id+1

    let fillRandomUniformMatrix4D (scaling_factor : float32) (target: d4M) =
        let weights_total_size = target.num_feature_maps*target.num_channels*target.num_rows*target.num_cols
        
        cudaRandom.Fill(stream_id,weights_total_size,target.dArray.Ptr,scaling_factor)
        stream_id <- stream_id+1

    let createRandomUniformMatrix weights_num_rows weights_num_cols (scaling_factor : float32) location =
        let weights_total_size = weights_num_rows*weights_num_cols
        
        let cudaBuffer = cudaRandom.AllocCUDAStreamBuffer weights_total_size
        cudaRandom.Fill(stream_id,weights_total_size,cudaBuffer,scaling_factor,location)
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

    let createRandomMatrix a b =
        let scale = 1.0f/sqrt((a) |> float32)
        let location = -scale*0.5f
        createRandomUniformMatrix a b scale location

    /// Creates an uninitialized empty matrix.
    let createEmpty4DMatrix num_feature_maps num_channels num_rows num_cols =
        {num_feature_maps=num_feature_maps; num_channels=num_channels; num_rows = num_rows; num_cols = num_cols; dArray = worker.Malloc<float32>(num_feature_maps*num_channels*num_rows*num_cols)} : d4M

    /// Create an empty matrix with the dimensions as the target.
    let createEmpty4DMatrixLike (target: d4DMatrix<'T>) =
        {num_feature_maps=target.num_feature_maps; num_channels=target.num_channels; num_rows = target.num_rows; num_cols = target.num_cols; dArray = worker.Malloc<'T>(target.num_feature_maps*target.num_channels*target.num_rows*target.num_cols)} : d4DMatrix<'T>

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
            if m <> C.num_rows || n <> C.num_cols then failwith "m <> C.num_rows || n <> C.num_cols in sgemm2"
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

  
    /// A top-k selection activation function. Unlike the last few attempts this one trully deserves the name
    /// of k-selection. Because row_size is determined statically and is a multiple of 32, 
    /// I've been able to unroll all the loops and store the variables into registers.

    /// 1.19 seconds per 10k iterations for the 1024x250 case. Is only 3x slower than map and roughly 50x faster
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


    /// Based on the winner-take-all autoencoder paper by Makhzhani.
    /// Is very fast and efficient, moreso that the k-sparse function.
    /// Allocates temporary storage for the transpose.
    /// http://arxiv.org/pdf/1409.2752.pdf
    type sparseWTAActivationModule(target, num_rows, num_cols, k, if_create_temp_storage) =
        inherit GPUModule(target)

        let grid_size = 384
        let block_size = 32

        let temporary_transpose_storage = 
            if if_create_temp_storage then 
                    createEmptyMatrix num_rows num_cols
                else
                    createEmptyMatrix 0 0

        new (num_rows, num_cols, k) = new sparseWTAActivationModule(GPUModuleTarget.Worker(worker), num_rows, num_cols, k, true)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (x:deviceptr<float32>) (y:deviceptr<float32>) =
            let inline butterflyWarpMax (value: float32) = 
                let inline max (a: float32) (b: float32) =
                    if a > b then a else b
                let v1 = max value (__shfl_xor value 16 32)
                let v2 = max v1 (__shfl_xor v1 8 32)
                let v3 = max v2 (__shfl_xor v2 4 32)
                let v4 = max v3 (__shfl_xor v3 2 32)
                max v4 (__shfl_xor v4 1 32)

            let num_vars = divup num_rows 32
            let vars = __local__.Array<float32>(num_vars)
            // Point block_start to where the column starts in the array.
            let mutable col = blockIdx.x
        
            while col < num_cols do
                // i is the variable index
                // Store the variables into registers.
                // The reason the num_rows is static and multiple of 32 is so I
                // can unroll this loop and guarantee that the registers will be used
                // instead of spilled to global memory.

                let mutable lower_bound, upper_bound = System.Single.MinValue, System.Single.MaxValue
                __unroll()
                for i=0 to num_vars-1 do
                    // idx is the absolute index in the array
                    let row = threadIdx.x + i*32
                    let idx = row + col * num_rows
                    if row < num_rows then
                        vars.[i] <- x.[idx]

                __unroll()
                for iters=1 to k do
                    __unroll()
                    for i=0 to num_vars-1 do
                        // idx is the absolute index in the array
                        let row = threadIdx.x + i*32
                        let idx = row + col * num_rows
                        if row < num_rows then
                            if vars.[i] < upper_bound && lower_bound < vars.[i] then
                                lower_bound <- vars.[i]
                    upper_bound <- (butterflyWarpMax lower_bound)
                    lower_bound <- System.Single.MinValue

                __unroll()
                for i=0 to num_vars-1 do
                    // idx is the absolute index in the array
                    let row = threadIdx.x + i*32
                    let idx = row + col * num_rows
                    if row < num_rows then
                        y.[idx] <- if vars.[i] < upper_bound then 0.0f else vars.[i]

                col <- col + gridDim.x

        member this.Apply((dmat: dM), (output: dM)) =
            if dmat.num_rows <> num_rows then failwith "dmat.num_rows <> num_rows sparseWTAActivationModule"
            if dmat.dArray.Length <> output.dArray.Length then failwith "dmat.dArray.Length <> output.dArray.Length in sparseWTAActivationModule"
            let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
            this.GPULaunch <@ this.Kernel @> lp dmat.dArray.Ptr output.dArray.Ptr
            output

        member this.Apply((dmat: dM)) =
            if dmat.num_rows <> num_rows then failwith "dmat.num_rows <> num_rows sparseWTAActivationModule"
            let output = createEmptyMatrixLike dmat
            let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
            this.GPULaunch <@ this.Kernel @> lp dmat.dArray.Ptr output.dArray.Ptr
            output

        member this.ApplyTranspose((dmat: dM), (output: dM)) =
            if dmat.num_cols <> num_rows then failwith "dmat.num_cols <> num_rows sparseWTAActivationModule"
            if dmat.num_rows <> num_cols then failwith "dmat.num_rows <> num_cols sparseWTAActivationModule"
            if dmat.dArray.Length <> output.dArray.Length then failwith "dmat.dArray.Length <> output.dArray.Length in sparseWTAActivationModule"
            let lp = LaunchParam(min grid_size dmat.num_cols, block_size)

            sgeam2 T T 1.0f dmat 0.0f dmat temporary_transpose_storage |> ignore
            this.GPULaunch <@ this.Kernel @> lp temporary_transpose_storage.dArray.Ptr temporary_transpose_storage.dArray.Ptr
            sgeam2 T T 1.0f temporary_transpose_storage 0.0f temporary_transpose_storage output |> ignore
        
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
            use z = worker.Malloc([|0.0f|])
            this.GPULaunch <@ this.Kernel @> lp n x y z.Ptr
            z.Gather().[0]

        member this.Apply (x: dMatrix<float32>, y: dMatrix<float32>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryMapReduceModule"
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
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceTrinaryMapReduceModule"
            if s.num_rows <> y.num_rows || s.num_cols <> y.num_cols then 
                failwith "s.num_rows <> y.num_rows || s.num_cols <> y.num_cols in DeviceTrinaryMapReduceModule"
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryMapReduceModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, s.dArray.Ptr)


    /// Quadrary transform module for applying maping functions to four identically sized arrays.
    /// Can be in-place or pointed to a destination.
    type DeviceQuadraryTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T -> 'T -> 'T>) =
        inherit GPUModule(target)

        new (op:Expr<'T -> 'T -> 'T -> 'T -> 'T>) =
            new DeviceQuadraryTransformModule<'T>(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (z:deviceptr<'T>) (s:deviceptr<'T>) (d:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                d.[i] <- __eval(op) x.[i] y.[i] z.[i] d.[i]
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>, s:deviceptr<'T>, d:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y z s d

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>, s: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceQuadraryTransformModule"
            if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
                failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceQuadraryTransformModule"
            if s.num_rows <> z.num_rows || s.num_cols <> z.num_cols then 
                failwith "s.num_rows <> z.num_rows || s.num_cols <> z.num_cols in DeviceQuadraryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length 
            then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length in DeviceTrinaryTransformModule"
            let d = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.dArray.Ptr, d.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = d}: dMatrix<'T>

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>, s: dMatrix<'T>, d: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceQuadraryTransformModule"
            if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
                failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceQuadraryTransformModule"
            if s.num_rows <> z.num_rows || s.num_cols <> z.num_cols then 
                failwith "s.num_rows <> z.num_rows || s.num_cols <> z.num_cols in DeviceQuadraryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length 
            then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length in DeviceTrinaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.dArray.Ptr, d.dArray.Ptr)
            d


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
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
                failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceTrinaryTransformModule"
            let s = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = s}: dMatrix<'T>

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>, s: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
                failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length in DeviceTrinaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.dArray.Ptr)
            s

        member this.Apply (x: dMatrix<'T> option, y: dMatrix<'T> option, z: dMatrix<'T> option, s: dMatrix<'T> option) =
            match x,y,z with
                | Some x, Some y, Some z -> 
                    match s with
                        | Some s -> Some (this.Apply(x,y,z,s))
                        | None -> Some (this.Apply(x,y,z))
                | _ -> z

    type DeviceTrinaryCoefTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T -> 'T -> 'T -> 'T -> 'T>) =
        inherit GPUModule(target)

        new (op:Expr<'T -> 'T -> 'T -> 'T -> 'T -> 'T -> 'T>) =
            new DeviceTrinaryCoefTransformModule<'T>(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) coef_x (x:deviceptr<'T>) coef_y (y:deviceptr<'T>) coef_z (z:deviceptr<'T>) (s:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                s.[i] <- __eval(op) coef_x x.[i] coef_y y.[i] coef_z z.[i]
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>, s:deviceptr<'T>, coef_x, coef_y, coef_z) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n coef_x x coef_y y coef_z z s

        member this.Apply (coef_x, x: dMatrix<'T>, coef_y, y: dMatrix<'T>, coef_z, z: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
                failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceTrinaryTransformModule"
            let s = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.Ptr, coef_x, coef_y, coef_z)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = s}: dMatrix<'T>

        member this.Apply (coef_x, x: dMatrix<'T>, coef_y, y: dMatrix<'T>, coef_z, z: dMatrix<'T>, s: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if z.num_rows <> y.num_rows || z.num_cols <> y.num_cols then 
                failwith "z.num_rows <> y.num_rows || z.num_cols <> y.num_cols in DeviceTrinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length || z.dArray.Length <> s.dArray.Length in DeviceTrinaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, s.dArray.Ptr, coef_x, coef_y, coef_z)
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
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
            let z = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = z}: dMatrix<'T>

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceBinaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr)
            z

        member this.Apply (x: dMatrix<'T> option, y: dMatrix<'T> option, z: dMatrix<'T> option) =
            match x,y with
                | Some x, Some y -> 
                    match z with
                        | Some z -> Some (this.Apply(x,y,z))
                        | None -> Some (this.Apply(x,y))
                | _ -> z

    type DeviceBinaryCoefTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T -> 'T -> 'T>) =
        inherit GPUModule(target)

        new (op:Expr<'T -> 'T -> 'T -> 'T -> 'T>) =
            new DeviceBinaryCoefTransformModule<'T>(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) coef_x (x:deviceptr<'T>) coef_y (y:deviceptr<'T>) (z:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                z.[i] <- __eval(op) coef_x x.[i] coef_y y.[i]
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>, coef_x, coef_y) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n coef_x x coef_y y z

        member this.Apply (coef_x, x: dMatrix<'T>, coef_y, y: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceBinaryTransformModule"
            let z = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr, coef_x, coef_y)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = z}: dMatrix<'T>

        member this.Apply (coef_x, x: dMatrix<'T>, coef_y, y: dMatrix<'T>, z: dMatrix<'T>) =
            if x.num_rows <> y.num_rows || x.num_cols <> y.num_cols then 
                failwith "x.num_rows <> y.num_rows || x.num_cols <> y.num_cols in DeviceBinaryTransformModule"
            if x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length || y.dArray.Length <> z.dArray.Length in DeviceBinaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr, coef_x, coef_y)
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

        member this.Apply(x: dMatrix<'T>) =
            let y = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = y}: dMatrix<'T>

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>) =
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceUnaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr)
            y

    type DeviceUnaryCoefTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T>) =
        inherit ILGPUModule(target)

        new (op:Expr<'T -> 'T -> 'T>) =
            new DeviceUnaryCoefTransformModule<'T>(GPUModuleTarget.Worker(worker), op)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) coef_x (x:deviceptr<'T>) (y:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                y.[i] <- __eval(op) coef_x x.[i] 
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, coef_x) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n coef_x x y 

        member this.Apply (coef_x, x: dMatrix<'T>) =
            let y = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.Ptr, coef_x)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = y}: dMatrix<'T>

        member this.Apply (coef_x, x: dMatrix<'T>, y: dMatrix<'T>) =
            if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in DeviceUnaryTransformModule"
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, coef_x)
            y

    /// Fills the array with coef_x.
    let setModule = new DeviceUnaryCoefTransformModule<float32> <@ fun coef_x _ -> coef_x @>

    type elementwiseMultiplyAndAverageModule(target, slice_size, groups_per_slice) =
        inherit GPUModule(target)

        /// Default slice size and the number of groups per slice is 32 and 8 respectively.
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
                    z.[row] <- alpha*mem.[threadIdx.x] + beta*z.[row]
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

            this.GPULaunch <@ this.Kernel @> lp y.num_rows y.num_cols (alpha / float32 y.num_cols) x.dArray.Ptr y.dArray.Ptr beta z.dArray.Ptr
            z

        member this.ElementwiseMultiplyAndAverage(alpha, x: dM, y: dM) =
            let z = createEmptyMatrix x.num_rows 1
            this.ElementwiseMultiplyAndAverage(alpha,x,y,0.0f,z)

    /// A module for elementwise multiplication with broadcasting.
    /// x is a nx1 matrix and y (and optionally z) are nxm matrices.
    /// This module's function calculates elemenwise x * y for every column of y and adds it to z.
    type broadcastingMultiplicationModule(target, slice_size, groups_per_slice) =
        inherit GPUModule(target)

        /// Default slice size and the number of groups per slice is 32 and 8 respectively.
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
                    // It adds directly to z
                    z.[idx] <- y.[idx]*elem+z.[idx]

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
            setModule.Apply(0.0f, z, z) |> ignore
            this.BroadcastMultiply(x,y,z)
    
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

        member this.Apply (x: d4DMatrix<'T>) =
            let blockSize = 256
            let gridSize = divup x.num_feature_maps blockSize
            let lp = LaunchParam(gridSize, blockSize)
            let dy = this.GPUWorker.Malloc<int>(x.num_feature_maps)
            this.GPULaunch <@ this.Kernel @> lp x.num_feature_maps x.dArray.Ptr dy.Ptr x.num_channels
            dy

    type maxNormRegularizationModule(target, num_rows) =
        inherit GPUModule(target)

        let grid_size = 384
        let block_size = 32

        new num_rows = new maxNormRegularizationModule(GPUModuleTarget.Worker(worker), num_rows)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (num_cols:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (norm_constant: float32) =
            let inline butterflyWarpReduce (value:float32) = 
                let v1 = value + __shfl_xor value 16 32
                let v2 = v1 + __shfl_xor v1 8 32
                let v3 = v2 + __shfl_xor v2 4 32
                let v4 = v3 + __shfl_xor v3 2 32
                v4 + __shfl_xor v4 1 32
            // Point block_start to where the column starts in the array.
            let mutable col = blockIdx.x
            let num_warps = divup num_rows 32

            while col < num_cols do
                let mutable acc = 0.0f
                __unroll()
                for i=0 to num_warps-1 do
                    // idx is the absolute index in the array
                    let row = threadIdx.x + i*32
                    if row < num_rows then
                        let idx = row + col * num_rows
                        acc <- acc + (x.[idx]*x.[idx])

                let norm_distance = sqrt(butterflyWarpReduce acc)

                // Apply max-norm regularization
                if norm_distance > norm_constant then
                    let norm = norm_constant/norm_distance
                    __unroll()
                    for i=0 to num_warps-1 do
                        // idx is the absolute index in the array
                        let row = threadIdx.x + i*32
                        if row < num_rows then
                            let idx = row + col * num_rows
                            // Rescale the weights
                            y.[idx] <- x.[idx]*norm

                col <- col + gridDim.x

        member this.Apply((dmat: dM), (rmat: dM), norm_constant) =
            if dmat.dArray.Length <> rmat.dArray.Length then failwith "dmat.dArray.Length <> rmat.dArray.Length in maxNormRegularizationModule"
            if dmat.num_rows <> num_rows then failwith "dmat.num_rows <> num_rows in maxNormRegularizationModule"
            let lp = LaunchParam(min grid_size dmat.num_rows, block_size)
            this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr rmat.dArray.Ptr norm_constant
            rmat

        member this.Apply((dmat: dM), norm_constant) =
            let lp = LaunchParam(min grid_size dmat.num_rows, block_size)
            let rmat = createEmptyMatrixLike dmat
            this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr rmat.dArray.Ptr norm_constant
            rmat


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
        {num_rows=num_rows; num_cols=l/num_rows; dArray=worker.Malloc(ar)}: dM

    let createEmptyAndSetZero (example: dM) =
        let t = createEmptyMatrixLike example
        setModule.Apply(0.0f, t, t)

    let createEmptyAndSetZero2 (example: dM option) =
        match example with
            | Some example ->
                let t = createEmptyMatrixLike example
                Some (setModule.Apply(0.0f, t, t))
            | None -> None

    // Dynamically allocates memory if the matrix has not been used before.
    let dynamic_multiply T1 T2 alpha weights input beta dest =
        match dest with
            | Some dest -> 
                match weights, input with
                    | Some weights, Some input ->
                        let t = sgemm2 T1 T2 alpha weights input !beta dest
                        beta := 1.0f
                        Some t
                    | _ -> Some dest
            | None -> 
                match weights, input with
                    | Some weights, Some input ->
                        let t = sgemm T1 T2 alpha weights input
                        beta := 1.0f
                        Some t
                    | _ -> dest

    // Dynamically allocates memory if the matrix has not been used before.
    let dynamic_add T1 T2 alpha grads beta weights dest =
        match dest with
            | Some dest -> 
                match grads, weights with
                    | Some grads, Some weights ->
                        let t = sgeam2 T1 T2 alpha grads beta weights dest
                        Some t
                    | _ -> Some dest
            | None -> 
                match grads, weights with
                    | Some grads, Some weights ->
                        let t = sgeam T1 T2 alpha grads beta weights
                        Some t
                    | _ -> dest