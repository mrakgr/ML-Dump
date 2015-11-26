// Works great. 3 epochs are quite enough for it to converge on Mnist with a batch size of 125, 
// learning rate 0.001f and momentum of 0.99f.

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

let train = make_imageset trainSetData trainSetLabels
let test = make_imageset testSetData testSetLabels


/// The Mnist training set split into batches of 250.
let batch_size = 125
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

let testing_batches =
    [|
    for i in 0..batch_size..test.num_images-1 do
        let s1 = test.num_rows*test.num_cols*i
        let s2 = test.num_rows*test.num_cols*(i+batch_size)-1
        let dtest_data: dM = 
                      {num_rows = test.num_rows * test.num_cols
                       num_cols = batch_size
                       dArray = worker.Malloc(test.float_data.[s1..s2])}

        if (dtest_data.num_cols*dtest_data.num_rows <> dtest_data.dArray.Length)
        then failwith "Invalid batch size (test)."

        let s1 = 10*i
        let s2 = 10*(i+batch_size)-1
        let dtest_label: dM =
                           {num_rows = 10
                            num_cols = batch_size
                            dArray = worker.Malloc(test.float_labels.[s1..s2])}
        if (dtest_label.num_cols*dtest_label.num_rows <> dtest_label.dArray.Length)
        then failwith "Invalid batch size (label)."
        yield (dtest_data, dtest_label)|]

// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

// Relu activation.
//let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a > 0.0f then a else 0.0f @>

// The k-sparse activation.
//let sparsePiecewiseLinearActivation = new sparsePiecewiseLinearActivationModule(1024,20)

// The WTA sparse activation.
let sparseWTAActivation = new sparseWTAActivationModule(batch_size,1024,6)
sparseWTAActivation.GPUForceLoad()

let sparse_wta_autoencoder weights learning_rate momentum_rate num_epochs (training_batches: (dM*dM) []) (testing_batches: (dM*dM) []) =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 
    let grad1 = createEmptyMatrixLike weights
    setModule.Apply(0.0f,grad1,grad1) |> ignore

    let z2 = sgemm nT nT 1.0f weights z1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let squared_cost_error2 = sgemm T nT 1.0f weights squared_cost_error
    
    let costSquaredError (batch: dM) weights =
    
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = sparseWTAActivation.ApplyTranspose(z1, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        squaredCostModule.Apply(batch, z2) * inv_batch_size
        

    let gradient (batch: dM) weights = 
        let inv_batch_size = 1.0f / float32 batch.num_cols

        // Nesterov's Momentum
        sgeam2 nT nT momentum_rate grad1 1.0f weights weights |> ignore

        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = sparseWTAActivation.ApplyTranspose(z1, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        
        let squared_cost_error2 = sgemm2 T nT 1.0f weights squared_cost_error 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore

        sgeam2 nT nT -momentum_rate grad1 1.0f weights weights |> ignore

        sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error a1 momentum_rate grad1 |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) batch squared_cost_error2 1.0f grad1 |> ignore

        sgeam2 nT nT 1.0f grad1 1.0f weights weights |> ignore

    let calculate_validation_error() =
        let mutable c = 0.0f
        for batch,l in testing_batches do
            c <- c + (costSquaredError batch weights)
        c / float32 testing_batches.Length
                
    printfn "Square error cost of the reconstruction (before optimization) is %f" (calculate_validation_error())

    for epoch=1 to num_epochs do
        for batch,_ in training_batches do
            gradient batch weights

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (calculate_validation_error())

(*
/// Logistic(x)
//let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>
let logisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> 
        if x <= 0.0f then 0.0f
        else if x >= 1.0f then 1.0f
        else x @>

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                let b_max = min 0.999999f b
                                let b_min = max 0.000001f b
                                a*(log b_min) + (1.0f-a)*log (1.0f - b_max)@>

let logistic_regression weights learning_rate num_epochs (training_batches: (dM*dM) []) (testing_batches: (dM*dM) []) = 
    let batch,_ = training_batches.[0]
    let inv_batch_size = 1.0f / float32 batch.num_cols

    let z1 = sgemm T nT 1.0f weights batch

    let z1_c = sgemm T nT 1.0f weights dtest_data 

    let costLogRegression (batch: dM) weights labels =
        let alpha = -1.0f/float32 batch.num_cols

        ///logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        let a1 = logisticActivationModule.Apply(z1, z1)

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
*)
(*
let feedforward_sparse_pass weights (training_batches: (dM*dM) []) dtest_data keepin_rate start_k =
    let training_batches_sparse =
        [|for batch,l in training_batches do
            let z1 = sgemm T nT keepin_rate weights batch
            sparsePiecewiseLinearActivation.Apply(z1, start_k, z1) |> ignore
            yield z1, l|]

    let dtest_data_sparse = sgemm T nT keepin_rate weights dtest_data
    sparsePiecewiseLinearActivation.Apply(dtest_data_sparse, start_k, dtest_data_sparse) |> ignore
    training_batches_sparse, dtest_data_sparse

let test_time (training_batches_sparse: (dM*dM) []) learning_rate epochs (dtest_data_sparse: dM) =
    let weights2 = createRandomUniformMatrix dtest_data_sparse.num_rows 10 1e-3f
    logistic_regression weights2 learning_rate epochs training_batches_sparse dtest_data_sparse

    let rowReducer = new maxRowReduceModule<float32>()

    let predictions = sgemm T nT 1.0f weights2 dtest_data_sparse
    let max_pred = rowReducer.Apply(predictions)
    let max_labels = rowReducer.Apply(dtest_label)

    let pr,l = max_pred.Gather(), max_labels.Gather()

    let mutable c = 0
    for i=0 to pr.Length-1 do
        if pr.[i] = l.[i] then c <- c + 1
    printfn "The accuracy is %i/%i" c pr.Length
    weights2
*)

let save_bitmap (weights: dM) =
    let num_rows_and_cols = sqrt(float weights.num_rows) |> int
    let bitmap = make_bitmap_from_imageset weights num_rows_and_cols num_rows_and_cols 40 25
    bitmap.Save(@"C:\!NN\wta_1g.bmp")

let weights = createRandomUniformMatrix 784 1024 1e-3f
sparse_wta_autoencoder weights 0.001f 0.99f 3 training_batches testing_batches

save_bitmap weights
