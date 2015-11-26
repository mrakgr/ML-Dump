// Record on Mnist: 98.36% using 1k epochs, 0.01f learning_rate and start_k of 700.

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
open Microsoft.FSharp.Quotations

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

let mutable dtest_data: dM = 
                {num_rows = test.num_rows*test.num_cols
                 num_cols = test.num_images
                 dArray = worker.Malloc(test.float_data)}

let dtest_label: dM =
                  {num_rows = 10
                   num_cols = test.num_images
                   dArray = worker.Malloc(test.float_labels)}

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

let hidden_layer_width = 1024

/// Logistic(x)
let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                let b_max = min 0.999999f b
                                let b_min = max 0.000001f b
                                a*(log b_min) + (1.0f-a)*log (1.0f - b_max)@>

let weights_l1 = createRandomUniformMatrix 784 hidden_layer_width 1e-5f
let weights_l2 = createRandomUniformMatrix (hidden_layer_width) 10 1e-5f

let sparse_network weights_l1 weights_l2 learning_rate num_epochs start_k =
    let batch,labels = training_batches.[0]
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Forward pass
    let z1 = sgemm T nT 1.0f weights_l1 batch 

    // Creates the sparsePiecewiseLinearActivationModule. I accidentaly the cost function.
    let sparseActivationModule = new sparsePiecewiseLinearActivationModule(z1.num_rows)

    let a1 = sparseActivationModule.Apply(z1,start_k, z1)
    let z2 = sgemm T nT 1.0f weights_l2 a1
    let a2 = logisticActivationModule.Apply(z2, z2)

    // Backprop for the 2nd layer
    let cross_entropy_error = binaryErrorModule.Apply(labels,a2)

    // Backprop for the 1st layer
    let sparse_layer_error = sgemm nT nT 1.0f weights_l2 cross_entropy_error

    let z1_c = sgemm T nT 1.0f weights_l1 dtest_data
    let z2_c = sgemm T nT 1.0f weights_l2 z1_c

    let costSquaredError batch weights_l1 weights_l2 labels =
        let alpha = -1.0f/float32 batch.num_cols
        // Forward pass
        let z1 = sgemm2 T nT 1.0f weights_l1 batch 0.0f z1_c
        let a1 = sparseActivationModule.Apply(z1,start_k, z1)
        let z2 = sgemm2 T nT 1.0f weights_l2 a1 0.0f z2_c
        let a2 = logisticActivationModule.Apply(z2, z2)
        alpha*crossEntropyCostModule.Apply(labels, a2)

    let gradient batch weights_l1 weights_l2 labels =
        // Forward pass
        let z1 = sgemm2 T nT 1.0f weights_l1 batch 0.0f z1
        let a1 = sparseActivationModule.Apply(z1,start_k,z1)
        let z2 = sgemm2 T nT 1.0f weights_l2 a1 0.0f z2
        let a2 = logisticActivationModule.Apply(z2, z2)

        // Backprop for the 2nd layer
        let cross_entropy_error = binaryErrorModule.Apply(labels,a2,cross_entropy_error)

        // Add directly to the weights.
        let weights_grad_l2 = sgemm2 nT T (-learning_rate*inv_batch_size) a1 cross_entropy_error 1.0f weights_l2

        // Backprop for the 1st layer
        let sparse_layer_error = sgemm2 nT nT 1.0f weights_l2 cross_entropy_error 0.0f sparse_layer_error
        binarySparseErrorModule.Apply(sparse_layer_error,a1,sparse_layer_error) |> ignore

        // Add directly to the weights.
        let weights_grad_l1 = sgemm2 nT T (-learning_rate*inv_batch_size) batch sparse_layer_error 1.0f weights_l1
        weights_grad_l1, weights_grad_l2

    printfn "Cross entropy error of the logistic regression layer before optimization is %f"  (costSquaredError dtest_data weights_l1 weights_l2 dtest_label)

    for epoch=1 to num_epochs do
        let current_k = start_k
        let mutable c = 0
        for batch,l in training_batches do
            gradient batch weights_l1 weights_l2 l |> ignore
            
        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (costSquaredError dtest_data weights_l1 weights_l2 dtest_label)

let start_k = 700

#time
sparse_network weights_l1 weights_l2 0.01f 1 start_k
#time

let rowReducer = new maxRowReduceModule<float32>()

let z1 = sgemm T nT 1.0f weights_l1 dtest_data
let sparseActivationModule = new sparsePiecewiseLinearActivationModule(z1.num_rows)
let a1 = sparseActivationModule.Apply(z1,start_k,z1)
let z2 = sgemm T nT 1.0f weights_l2 a1

let max_pred = rowReducer.Apply(z2)
let max_labels = rowReducer.Apply(dtest_label)

let pr,l = max_pred.Gather(), max_labels.Gather()

let mutable c = 0
for i=0 to pr.Length-1 do
    if pr.[i] = l.[i] then c <- c + 1
printfn "The accuracy is %i/%i" c pr.Length