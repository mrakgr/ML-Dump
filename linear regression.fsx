// This one does 85.1% on Mnist vs 91.5% for logistic regression (without the bias term).
// 85.5% with the bias term. 86% if I quadruple the number of epochs and cut the learning
// rate by 3/4ths.

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

//let weights = createRandomUniformMatrix 2 5 1e-0f
//let bias_input = createRandomUniformMatrix 2 1 1e-0f

let train = make_imageset trainSetData trainSetLabels
let test = make_imageset testSetData testSetLabels

/// The Mnist training set split into batches of 250.
let batch_size = 250
let inv_batch_size = 1.0f / float32 batch_size
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

let sumReduce = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

let distanceModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

let linear_autoencoder learning_rate num_epochs =
    let weights = createRandomUniformMatrix 784 10 1e-5f
    let biases = createRandomUniformMatrix 10 1 1e-5f
    let batch, l = training_batches.[0]

    /// Prealocate memory.
    let z1 = sgemm T nT 1.0f weights batch 
    let squared_cost_error = distanceModule.Apply(l, z1)
    let grad_first_layer = sgemm nT T inv_batch_size batch squared_cost_error

    let ones = createEmptyMatrix batch.num_cols 1
    onesModule.Apply(ones, ones) |> ignore

    let bias_grad = sgemv nT inv_batch_size squared_cost_error ones

    let z1_c = sgemm T nT 1.0f weights dtest_data

    let costSquaredError batch weights labels =
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        addBias z1 biases
        sumReduce.Apply(labels, z1) / float32 batch.num_cols

    let gradient batch weights biases labels = 
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        addBias z1 biases

        /// -(y-a) = a-y
        /// This is the error term in the last layer...and in the following layers
        let squared_cost_error = distanceModule.Apply(labels, z1, squared_cost_error)

        let batch_grad = sgemm2 nT T inv_batch_size batch squared_cost_error 0.0f grad_first_layer
        let bias_grad = sgemv2 nT inv_batch_size squared_cost_error ones 0.0f bias_grad
        batch_grad, bias_grad

    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data weights dtest_label)

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            let batch_grad, bias_grad = gradient batch weights biases l
            // Add them to the weights.
            sgeam2 nT nT 1.0f weights (-learning_rate) batch_grad weights |> ignore
            sgeam2 nT nT 1.0f biases (-learning_rate) bias_grad biases |> ignore

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data weights dtest_label)
    weights, biases
        
#time
let weights, biases = linear_autoencoder 0.0025f 500
#time

let rowReducer = new maxRowReduceModule<float32>()

let predictions = sgemm T nT 1.0f weights dtest_data
addBias predictions biases
let max_pred = rowReducer.Apply(predictions)
let max_labels = rowReducer.Apply(dtest_label)

let pr,l = max_pred.Gather(), max_labels.Gather()

let mutable c = 0
for i=0 to pr.Length-1 do
    if pr.[i] = l.[i] then c <- c + 1
printfn "The accuracy is %i/%i" c pr.Length

