// Record: 96.04%. epochs=800. 96.11%. epochs = 1200. (for the max autoencoder)
// Record: 96.31%. epochs=600. (for the min autoencoder)
// ...It is crap now that I've fixed the bug.

#load "load_mnist.fsx"
open Load_mnist.MnistLoad
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

/// An extremely fast activation function. This one trully deserves the name
/// of k-selection. Because row_size is determined statically and is a multiple of 32, 
/// I've been able to unroll all the loops and store the variables into registers.

/// 1.19 seconds per 10k iterations. Is only 3x slower than map and roughly 50x faster
/// than sort.

/// On Maxwell cards having the small block size of 32 is very efficient.
/// http://arxiv.org/abs/1312.5663
type clustering_1_32_PiecewiseLinearActivationModule(target, num_rows) =
    inherit GPUModule(target)

    let _ = if num_rows % 32 <> 0 then failwith "num_rows has to be a multiple of 32 in sparsePiecewiseLinearActivationModule"

    let grid_size = 384
    let block_size = 32

    /// Default number of splits=20
    new num_rows = new clustering_1_32_PiecewiseLinearActivationModule(GPUModuleTarget.Worker(worker), num_rows)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (num_cols:int) (x:deviceptr<float32>) (y:deviceptr<float32>) =
        let inline butterflyWarpMax (value:float32) = 
            let v = __shfl_xor value 16 32
            let maxval1 = max v value
            let v1 = __shfl_xor v 8 32
            let maxval2 = max v1 maxval1
            let v2 = __shfl_xor v1 4 32
            let maxval3 = max v2 maxval2
            let v3 = __shfl_xor v2 2 32
            let maxval4 = max v3 maxval3
            let v4 = __shfl_xor v3 1 32
            let maxval5 = max v4 maxval4
            maxval5

        let num_vars = num_rows/__warp_size()
        // Point block_start to where the column starts in the array.
        let mutable col = blockIdx.x

        while col < num_cols do
            // i is the variable index
            // Store the variables into registers.
            // The reason the num_rows is static and multiple of 32 is so I
            // can unroll this loop and guarantee that the registers will be used
            // instead of spilled to global memory.
            __unroll()
            for i=0 to num_vars-1 do
                // idx is the absolute index in the array
                let idx = threadIdx.x + i*32 + col * num_rows
                let vars = x.[idx]
                let tmax = butterflyWarpMax vars

                // Let the function activate if it is above the threshold.
                y.[idx] <- if vars <= tmax then vars else 0.0f

            col <- col + gridDim.x

    member this.Apply((dmat: dM), (output: dM)) =
        if dmat.num_rows <> num_rows then failwith "dmat.num_rows <> num_rows sparsePiecewiseLinearActivationModule"
        if dmat.dArray.Length <> output.dArray.Length then failwith "dmat.dArray.Length <> output.dArray.Length in sparsePiecewiseLinearActivationModule"
        let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
        this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr output.dArray.Ptr
        output

    member this.Apply((dmat: dM)) =
        if dmat.num_rows <> num_rows then failwith "dmat.num_rows <> num_rows sparsePiecewiseLinearActivationModule"
        let output = createEmptyMatrixLike dmat
        let lp = LaunchParam(min grid_size dmat.num_cols, block_size)
        this.GPULaunch <@ this.Kernel @> lp dmat.num_cols dmat.dArray.Ptr output.dArray.Ptr
        output

let train = make_imageset trainSetData trainSetLabels
let test = make_imageset testSetData testSetLabels


/// The Mnist training set split into batches of 250.
let batch_size = 100
let training_batches =
    [|
    for i in 0..batch_size..train.num_images-1 do
        let s1 = train.num_rows*train.num_cols*i
        let s2 = train.num_rows*train.num_cols*(i+batch_size)-1
        let dtrain_data: dM = 
                      {num_rows = train.num_rows*train.num_cols
                       num_cols = batch_size
                       dArray = worker.Malloc(train.float_data.[s1..s2])}

        if (dtrain_data.num_cols*dtrain_data.num_rows <> dtrain_data.dArray.Length)
        then failwith "Invalid batch size (test)."

        let s1 = 10*i
        let s2 = 10*(i+batch_size)-1
        let dtrain_label: dM =
                           {num_rows = 10
                            num_cols = batch_size
                            dArray = worker.Malloc(train.float_labels.[s1..s2])}
        if (dtrain_label.num_cols*dtrain_label.num_rows <> dtrain_label.dArray.Length)
        then failwith "Invalid batch size (label)."
        yield (dtrain_data, dtrain_label)|]

let dtest_data: dM = 
                {num_rows = test.num_rows*test.num_cols
                 num_cols = test.num_images
                 dArray = worker.Malloc(test.float_data)}

let dtest_label: dM =
                  {num_rows = 10
                   num_cols = test.num_images
                   dArray = worker.Malloc(test.float_labels)}

// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

let clustering_autoencoder weights learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 

    // The sparse activation function module inspired by the k-sparse autoencoder.
    // http://arxiv.org/abs/1312.5663
    let sparseActivationModule = new clustering_1_32_PiecewiseLinearActivationModule(z1.num_rows)

    let z2 = sgemm nT nT 1.0f weights z1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let squared_cost_error2 = sgemm T nT 1.0f weights squared_cost_error 
    
    // Memory for the cost function of the test set.
    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c

    let costSquaredError batch weights =
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c 
        let a1 = sparseActivationModule.Apply(z1, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) * inv_batch_size

    let gradient batch weights = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = sparseActivationModule.Apply(z1,z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error a1 1.0f weights |> ignore
        let squared_cost_error2 = sgemm2 T nT 1.0f weights squared_cost_error 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) batch squared_cost_error2 1.0f weights |> ignore
                
    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data weights)

    for epoch=1 to num_epochs do
        for batch,_ in training_batches do
            gradient batch weights

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data weights)

/// Logistic(x)
let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                let b_max = min 0.999999f b
                                let b_min = max 0.000001f b
                                a*(log b_min) + (1.0f-a)*log (1.0f - b_max)@>

let logistic_regression weights learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data = 
    let batch,_ = training_batches.[0]
    let inv_batch_size = 1.0f / float32 batch.num_cols

    let z1 = sgemm T nT 1.0f weights batch

    let z1_c = sgemm T nT 1.0f weights dtest_data 

    let costLogRegression batch weights labels =
        let alpha = -1.0f/float32 batch.num_cols

        ///logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        let a1 = logisticActivationModule.Apply(z1,z1)

        let cross_entropy_cost = alpha * crossEntropyCostModule.Apply(labels, a1)
        
        cross_entropy_cost

    let gradient batch weights labels =
        /// logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = logisticActivationModule.Apply(z1,z1)
        /// -(labels-a1)
        let cross_entropy_error = binaryErrorModule.Apply(labels,a1,a1)
        /// Add directly to the weights.
        sgemm2 nT T (-learning_rate*inv_batch_size) batch cross_entropy_error 1.0f weights |> ignore

    printfn "Cross entropy error of the logistic regression layer before optimization is %f" (costLogRegression dtest_data weights dtest_label)

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            gradient batch weights l

        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (costLogRegression dtest_data weights dtest_label)

let feedforward_sparse_pass weights (training_batches: (dM*dM) []) dtest_data =
    let batch, l = training_batches.[0]
    let z1 = sgemm T nT 1.0f weights batch
    let sparseActivationModule = new clustering_1_32_PiecewiseLinearActivationModule(z1.num_rows)

    let training_batches_sparse =
        [|for batch,l in training_batches do
            let z1 = sgemm T nT 1.0f weights batch
            sparseActivationModule.Apply(z1,z1) |> ignore
            yield z1, l|]

    let dtest_data_sparse = sgemm T nT 1.0f weights dtest_data
    sparseActivationModule.Apply(dtest_data_sparse,dtest_data_sparse) |> ignore
    training_batches_sparse, dtest_data_sparse

let weights = createRandomUniformMatrix 784 1024 1e-4f

clustering_autoencoder weights 0.01f 20 training_batches dtest_data

let training_batches_sparse, dtest_data_sparse = feedforward_sparse_pass weights training_batches dtest_data

let weights2 = createRandomUniformMatrix 1024 10 1e-3f

logistic_regression weights2 0.1f 300 training_batches_sparse dtest_data_sparse

let test_time() =
    let rowReducer = new maxRowReduceModule<float32>()

    let predictions = sgemm T nT 1.0f weights2 dtest_data_sparse
    let max_pred = rowReducer.Apply(predictions)
    let max_labels = rowReducer.Apply(dtest_label)

    let pr,l = max_pred.Gather(), max_labels.Gather()

    let mutable c = 0
    for i=0 to pr.Length-1 do
        if pr.[i] = l.[i] then c <- c + 1
    printfn "The accuracy is %i/%i" c pr.Length
test_time()

let bitmap = make_bitmap_from_imageset weights 28 28 40 25
bitmap.Save(@"C:\!NN\min_clustering_1_32_1.bmp")

