// Record: 96.6%. 300 epochs, 0.01f learning rate, 0 lambda.

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
let batch_size = 250
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

// For errors in the middle layers using Relu activations.
let binaryReluErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

// Relu activation.
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a >= 0.0f then a else 0.0f @>
//let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> max a 0.0f @>

let relu_autoencoder weights learning_rate lambda num_epochs (training_batches: (dM*dM) []) dtest_data =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    let l1RegularizationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> a-inv_batch_size*lambda*learning_rate*(if a >= 0.0f then 1.0f else -1.0f)@>

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 
    let z2 = sgemm nT nT 1.0f weights z1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let squared_cost_error2 = sgemm T nT 1.0f squared_cost_error weights
    
    // Memory for the cost function of the test set.
    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c

    let costSquaredError batch weights =
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c 
        let a1 = reluActivationModule.Apply(z1, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) * inv_batch_size

    let gradient batch weights = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = reluActivationModule.Apply(z1,z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error a1 1.0f weights |> ignore
        let squared_cost_error2 = sgemm2 T nT 1.0f squared_cost_error weights 0.0f squared_cost_error2
        binaryReluErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
        sgemm2 nT nT (-inv_batch_size*learning_rate) batch squared_cost_error2 1.0f weights |> ignore

        if lambda <> 0.0f then l1RegularizationModule.Apply(weights, weights) |> ignore
                
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

let feedforward_relu_pass weights (training_batches: (dM*dM) []) dtest_data =
    let training_batches_sparse =
        [|for batch,l in training_batches do
            let z1 = sgemm T nT 1.0f weights batch
            reluActivationModule.Apply(z1,z1) |> ignore
            yield z1, l|]

    let dtest_data_relu = sgemm T nT 1.0f weights dtest_data
    reluActivationModule.Apply(dtest_data_relu,dtest_data_relu) |> ignore
    training_batches_sparse, dtest_data_relu

let weights = createRandomUniformMatrix 784 1024 1e-4f
relu_autoencoder weights 0.01f 0.0f 10 training_batches dtest_data

let training_batches_relu, dtest_data_relu = feedforward_relu_pass weights training_batches dtest_data

let weights2 = createRandomUniformMatrix 1024 10 1e-3f
logistic_regression weights2 0.1f 500 training_batches_relu dtest_data_relu

let rowReducer = new maxRowReduceModule<float32>()

let predictions = sgemm T nT 1.0f weights2 dtest_data_relu
let max_pred = rowReducer.Apply(predictions)
let max_labels = rowReducer.Apply(dtest_label)

let pr,l = max_pred.Gather(), max_labels.Gather()

let mutable c = 0
for i=0 to pr.Length-1 do
    if pr.[i] = l.[i] then c <- c + 1
printfn "The accuracy is %i/%i" c pr.Length

let bitmap = make_bitmap_from_imageset weights 28 28 40 25
bitmap.Save(@"C:\!NN\relu_1.bmp")