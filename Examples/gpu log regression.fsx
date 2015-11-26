// This code is crap. A much better logistic regression implementation can be found in
// pretrained 2-layer deep net.

#I @"C:\F# Packages\packages\MathNet.Numerics.FSharp.3.7.0\lib\net40"
#I @"C:\F# Packages\packages\MathNet.Numerics.Data.Text.3.1.1\lib\net40"
#I @"C:\F# Packages\packages\MathNet.Numerics.3.7.0\lib\net40"
#r @"MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.dll"
#r @"MathNet.Numerics.Data.Text.dll"
#r @"MathNet.Numerics.dll"
#load "Types.fs"

#I @"C:\F# Packages\packages\Alea.CUDA.2.1.2.3274\private"
#I @"C:\F# Packages\packages\Alea.CUDA.2.1.2.3274\lib\net40"
#I @"C:\F# Packages\packages\Alea.CUDA.IL.2.1.2.3274\lib\net40"
#I @"C:\F# Packages\packages\Alea.CUDA.Unbound.2.1.2.3274\lib\net40"
#r @"Alea.CUDA.Unbound.dll"
#r @"Alea.CUDA.IL.dll"
#r @"Alea.CUDA.dll"
#r @"Alea.CUDA.CT.Native.X86.B64.Windows.dll"
#r "System.Configuration.dll"

open System
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.CULib
open Alea.CUDA.IL
open Alea.CUDA.Unbound.Rng
open Alea.CUDA.Unbound
open Alea.CUDA.Unbound.LinAlg.Matrix.Multiply.CUDA
open FSharp.Quotations

open Mnist.Types

open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

let worker = Worker.Default
let blob = new Blob(worker)
let cublas = CUBLAS.Default
let GPUMultiplication32 = DefaultMatrixMultiplyModuleF32.DefaultInstance

Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- @"C:\F# Packages\packages\Alea.CUDA.2.1.2.3274\private"

type LogisticRegParams = {
    num_trials : int
    num_iterations_per_trial : int
    learning_rate: float32
    lambda : float32
    }

let train = make_imageset trainSetData trainSetLabels
let test = make_imageset testSetData testSetLabels

#load "transform.fsx"
open Transform

type dM = Transform.dMatrix<float32>

let dtrain_data: dM = 
                  {num_rows = train.matrix.RowCount
                   num_cols = train.matrix.ColumnCount
                   dArray = worker.Malloc(train.matrix.ToColumnWiseArray())}

let dtrain_label: dM =
                   {num_rows = train.label_matrix.RowCount
                    num_cols = train.label_matrix.ColumnCount
                    dArray = worker.Malloc(train.label_matrix.ToColumnWiseArray())}

let dtest_data: dM = 
                {num_rows = test.matrix.RowCount
                 num_cols = test.matrix.ColumnCount
                 dArray = worker.Malloc(test.matrix.ToColumnWiseArray())}

let dtest_label: dM =
                  {num_rows = test.label_matrix.RowCount
                   num_cols = test.label_matrix.ColumnCount
                   dArray = worker.Malloc(test.label_matrix.ToColumnWiseArray())}

///Not transpose.
let nT = cublasOperation_t.CUBLAS_OP_N
///Transpose.
let T = cublasOperation_t.CUBLAS_OP_T

let sgemm2(transa, transb, (alpha: float32), (A:dM), (B:dM), beta, (C:dM)) =
        let a_col = if transa = nT then A.num_cols else A.num_rows
        let b_row = if transb = nT then B.num_rows else B.num_cols
        if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" A.num_cols A.num_rows)
        let m = if transa = nT then A.num_rows else A.num_cols
        let n = if transb = nT then B.num_cols else B.num_rows
        let k = a_col

        let lda = if transa = nT then m else k
        let ldb = if transb = nT then k else n
        let ldc = m

        cublas.Sgemm(transa, transb, m, n, k, alpha, A.dArray.Ptr, lda, B.dArray.Ptr, ldb, beta, C.dArray.Ptr, ldc)
        {num_rows=m; num_cols=n; dArray=C.dArray}: dM

let sgemm(transa, transb, (alpha: float32), (A:dM), (B:dM)) =
        let a_col = if transa = nT then A.num_cols else A.num_rows
        let b_row = if transb = nT then B.num_rows else B.num_cols
        if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" A.num_cols A.num_rows)
        let m = if transa = nT then A.num_rows else A.num_cols
        let n = if transb = nT then B.num_cols else B.num_rows
        let k = a_col

        let lda = if transa = nT then m else k
        let ldb = if transb = nT then k else n
        let ldc = m

        let C_dArray = worker.Malloc<float32>(m*n)
        cublas.Sgemm(transa, transb, m, n, k, alpha, A.dArray.Ptr, lda, B.dArray.Ptr, ldb, 0.0f, C_dArray.Ptr, ldc)
        {num_rows=m; num_cols=n; dArray=C_dArray}: dM

let createRandomUniformMatrix weights_num_rows weights_num_cols (scaling_factor : float32) location =
    let weights_total_size = weights_num_rows*weights_num_cols

    let cudaRandom = XorShift7.CUDA.DefaultUniformRandomModuleF32.Default.Create(1,1,uint32 DateTime.Now.Millisecond) :> IRandom<float32>
    let cudaBuffer = cudaRandom.AllocCUDAStreamBuffer weights_total_size
    cudaRandom.Fill(0,weights_total_size,cudaBuffer,scaling_factor,location*scaling_factor/2.0f)

    {num_rows = weights_num_rows; num_cols = weights_num_cols; dArray = cudaBuffer}:dM
    
let makeReduce (op:Expr<'T -> 'T -> 'T>)  =
    let compileReductionKernel (op:Expr<'T -> 'T -> 'T>) =
        worker.LoadProgram(
                        DeviceReduceImpl.DeviceReduce(op, worker.Device.Arch, PlatformUtil.Instance.ProcessBitness).Template
                        )

    let prog = compileReductionKernel op

    let runReduceProgram (sumProg : Program<DeviceReduceImpl.IDeviceReduceFactory<'A>>) (x: DeviceMemory<'A>) = 
        sumProg.Entry.Create(blob, x.Length)
               .Reduce(None, x.Ptr, x.Length)

    let reduceProg (x: DeviceMemory<'T>) = runReduceProgram prog x
    reduceProg

/// In-place version. C should either be A or B. 
let sgeam2(transa, transb, (alpha: float32), (A:dM), beta, (B:dM), (C:dM)) =

        let a_row = if transa = nT then A.num_rows else A.num_cols
        let a_col = if transa = nT then A.num_cols else A.num_rows
        let b_row = if transb = nT then B.num_rows else B.num_cols
        let b_col = if transb = nT then B.num_cols else B.num_rows
        
        if a_row <> b_row then failwith (sprintf "a_row <> b_row in sgemm! %i <> %i" a_row b_row)
        if a_col <> b_col then failwith (sprintf "a_col <> b_col in sgemm! %i <> %i" a_col b_col)

        let lda = if transa = nT then a_row else a_col
        let ldb = if transa = nT then b_row else b_col
        let ldc = a_row

        cublas.Sgeam(transa, transb, a_row, a_col, alpha, A.dArray.Ptr, lda, beta, B.dArray.Ptr, ldb, C.dArray.Ptr, ldc)
        {num_rows=a_row; num_cols=a_col; dArray=C.dArray}: dM

let inline sgeam(transa, transb, (alpha: float32), (A:dM), beta, (B:dM)) =
        let C_dArray = worker.Malloc<float32>(A.num_cols*A.num_rows)
        let C = {num_rows=0; num_cols=0; dArray=C_dArray}: dM
        sgeam2(transa, transb, (alpha: float32), (A:dM), beta, (B:dM), C)

let unboundsgemm(transa, transb, (alpha: float32), (A:dM), (B:dM)) =
    let a_row = if transa = nT then A.num_rows else A.num_cols
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    let b_col = if transb = nT then B.num_cols else B.num_rows

    let transa = if transa = nT then NoTranspose else Transpose
    let transb = if transb = nT then NoTranspose else Transpose

    if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" A.num_cols A.num_rows)

    let C_dArray = worker.Malloc<float32>(a_row*b_col)

    let gpuOutput = GPUMultiplication32.Mult(PrefetchingData, transa, transb, ColumnMajor, A.num_rows, A.num_cols, B.num_rows, B.num_cols, alpha, 0.0f, A.dArray.Ptr, B.dArray.Ptr, C_dArray.Ptr)
    {num_rows=a_row; num_cols=b_col; dArray=C_dArray}: dM

let unboundsgemm2(transa, transb, (alpha: float32), (A:dM), (B:dM), beta, (C:dM)) =
    let a_row = if transa = nT then A.num_rows else A.num_cols
    let a_col = if transa = nT then A.num_cols else A.num_rows
    let b_row = if transb = nT then B.num_rows else B.num_cols
    let b_col = if transb = nT then B.num_cols else B.num_rows

    let transa = if transa = nT then NoTranspose else Transpose
    let transb = if transb = nT then NoTranspose else Transpose

    if a_col <> b_row then failwith (sprintf "a_col <> b_row in sgemm! %i <> %i" A.num_cols A.num_rows)

    let gpuOutput = GPUMultiplication32.Mult(PrefetchingData, transa, transb, ColumnMajor, A.num_rows, A.num_cols, B.num_rows, B.num_cols, alpha, beta, A.dArray.Ptr, B.dArray.Ptr, C.dArray.Ptr)
    {num_rows=a_row; num_cols=b_col; dArray=C.dArray}: dM

/// For me, the cuBLAS sgemm is significantly faster than the Unbound's implementation.
/// I had not expected that preallocating memory would have such a great effect.
/// i5-4790k, GTX 970.
/// unboundsgemm = 12.75s
/// unboundsgemm2 = 7.3-7.84s (depending on the number of GCs)
/// sgemm = 9.63s
/// sgemm2 = 5.15s
(*
/// t is the preallocated memory matrix.
let t = sgemm(T,nT,1.0f,dWeights,dtrain_data)

#time
for i=1 to 3000 do
    unboundsgemm(T,nT,1.0f,dWeights,dtrain_data) |> ignore
#time

#time
for i=1 to 3000 do
    unboundsgemm2(T,nT,1.0f,dWeights,dtrain_data,0.0f,t) |> ignore
#time

#time
for i=1 to 3000 do
    sgemm(T,nT,1.0f,dWeights,dtrain_data) |> ignore
#time

#time
for i=1 to 3000 do
    sgemm2(T,nT,1.0f,dWeights,dtrain_data,0.0f,t) |> ignore
#time
*)

let logistic_regression (dtrain_data: dM) (dtrain_label: dM) dtest_data dtest_label (p: LogisticRegParams) = 

    let dWeights = createRandomUniformMatrix dtrain_data.num_rows dtrain_label.num_rows 1e-3f -0.5f

    /// Logistic(x). Inplace transform.
    let logistic_trans = new Transform.UnaryDeviceInPlaceTransformModule<float32>(GPUModuleTarget.DefaultWorker, 
                                 <@ fun x -> 1.0f/(1.0f+exp(-x)) @>)

    /// a*(log b) + (1.0f-a)*(1.0f - log b)
    let elemmult_trans = new Transform.DeviceBinaryTransformModule<float32>(GPUModuleTarget.DefaultWorker, 
                                <@ fun a b -> 
                                a*(log b) + (1.0f-a)*log (1.0f - b)@>)

    /// Reduce sum (float32) device function.
    let sumReduce: DeviceMemory<float32> -> float32 = makeReduce <@ fun (a:float32) b -> a + b @>

    /// x*x
    let pow2_trans = new Transform.UnaryDeviceTransformModule<float32>(GPUModuleTarget.DefaultWorker, 
                         <@ fun x -> x*x @>)

    let make_out_label_module cons =
            new Transform.DeviceBinaryTransformModule<float32>(GPUModuleTarget.DefaultWorker, 
                             <@ fun a b -> 
                             (a-b)*cons @>)

    /// (a - b)/num_batches for the training set.
    let out_label_train = make_out_label_module (1.0f/float32 dtrain_data.num_cols)

    let calculate_cost (data: dM) labels weights =
        let alpha = -1.0f/float32 data.num_cols

        ///logistic(dWeights.T*dtrain_data)
        let output = logistic_trans.Apply( sgemm(T,nT,1.0f,weights,data) )

        /// labels*(log output) + (1.0f-labels)*log (1.0f - output)
        let label_elemwise_mult_log_output = elemmult_trans.Apply(labels,output)
        
        /// alpha * sumall(label_elemwise_mult_log_output)
        let cross_entropy_cost = alpha*(sumReduce label_elemwise_mult_log_output.dArray)       
        
        /// x*x
        let dWeights_pow2 = pow2_trans.Apply(dWeights)

        /// alpha * p.lambda * sumall(x*x)
        let reg_cost = alpha*p.lambda*(sumReduce dWeights_pow2.dArray)
        cross_entropy_cost+reg_cost

    let calculate_grad data labels weights (out_label_trans: Transform.DeviceBinaryTransformModule<float32>) =
        /// logistic(dWeights.T*dtrain_data)
        let output = logistic_trans.Apply( sgemm(T,nT,1.0f,weights,data) )
        /// (output-label)*alpha
        let output_minus_label_times_alpha = out_label_trans.Apply(output, labels)
        /// data * ((output-label)*alpha).T
        let weights_grad = sgemm(nT,T,1.0f,data,output_minus_label_times_alpha)
        let float_num_batches = float32 data.num_cols
        let weights_reg_const = 2.0f*p.lambda/float_num_batches
        sgeam2(nT,nT,1.0f,weights_grad,weights_reg_const,weights,weights_grad)
        
    for trial=1 to p.num_trials do
        for iteration=1 to p.num_iterations_per_trial do
            let grad = calculate_grad dtrain_data dtrain_label dWeights out_label_train
            sgeam2(nT,nT,1.0f,dWeights,-p.learning_rate,grad,dWeights) |> ignore
        let cost = calculate_cost dtest_data dtest_label dWeights
        printfn "The cross entropy cost on the test set is %f at trial %i" cost trial

    dWeights

let p = {num_trials=200; num_iterations_per_trial=300; learning_rate=0.2f; lambda=0.1f}
#time
let opt_weights = logistic_regression dtrain_data dtrain_label dtest_data dtest_label p
#time

let rowReducer = new Transform.rowReduceModule<float32>(GPUModuleTarget.DefaultWorker,10)

let predictions = sgemm(T,nT,1.0f,opt_weights,dtest_data)
let max_pred = rowReducer.Apply predictions
let max_labels = rowReducer.Apply dtest_label

let pr,l = max_pred.Gather(), max_labels.Gather()

let mutable c = 0
for i=0 to pr.Length-1 do
    if pr.[i] = l.[i] then c <- c + 1
printfn "The accuracy is %i/%i" c pr.Length


    