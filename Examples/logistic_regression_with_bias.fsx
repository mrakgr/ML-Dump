// 91.44%.

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

/// Logistic(x)
let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                a*(log b) + (1.0f-a)*log (1.0f - b)@>

/// sumall(map(x*x)) for the L2 penalty
let squaredMapSum = new DeviceUnaryMapReduceModule <@ fun x -> x*x @>

let logistic_regression weights biases learning_rate lambda num_epochs = 
    let batch,_ = training_batches.[0]
    let inv_batch_size = 1.0f / float32 batch.num_cols

    let z1 = sgemm T nT 1.0f weights batch
    let weights_grad = sgemm nT T inv_batch_size batch z1

    let ones = createEmptyMatrix batch.num_cols 1
    onesModule.Apply(ones, ones) |> ignore

    let bias_grad = sgemv nT inv_batch_size z1 ones

    let z1_c = sgemm T nT 1.0f weights dtest_data 

    let costLogRegression batch weights biases labels =
        let alpha = -1.0f/float32 batch.num_cols

        ///logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        addBias z1 biases

        let a1 = logisticActivationModule.Apply(z1,z1)

        /// alpha * sumall(labels*(log output) + (1.0f-labels)*log (1.0f - output))
        let cross_entropy_cost = alpha * crossEntropyCostModule.Apply(labels, a1)
        
        /// alpha * p.lambda * sumall(x*x)
        let reg_cost = alpha*lambda*squaredMapSum.Apply(weights)
        cross_entropy_cost+reg_cost

    let gradient batch weights biases labels =
        let alpha = -1.0f/float32 batch.num_cols
        /// logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        addBias z1 biases
        let a1 = logisticActivationModule.Apply(z1,z1)
        /// -(labels-a1)
        let cross_entropy_error = binaryErrorModule.Apply(labels,a1)
        /// data * cross_entropy_error.T
        let weights_grad = sgemm2 nT T inv_batch_size batch cross_entropy_error 0.0f weights_grad
        // Constant for the L2 penalty.
        let weights_reg_const = 2.0f*lambda*inv_batch_size
        // Add the L2 penalty to the gradient.
        sgeam2 nT nT 1.0f weights_grad weights_reg_const weights weights_grad |> ignore

        let bias_grad = sgemv2 nT inv_batch_size cross_entropy_error ones 0.0f bias_grad
        weights_grad, bias_grad

    printfn "Cross entropy error of the logistic regression layer before optimization is %f" (costLogRegression dtest_data weights biases dtest_label)

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            let grad, bias_grad = gradient batch weights biases l
            // Add the gradients to the weights.
            sgeam2 nT nT 1.0f weights (-learning_rate) grad weights |> ignore
            sgeam2 nT nT 1.0f biases (-learning_rate) bias_grad biases |> ignore

        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (costLogRegression dtest_data weights biases dtest_label)

let weights = createRandomUniformMatrix 784 10 1e-5f
let biases = createRandomUniformMatrix 10 1 1e-5f
#time
logistic_regression weights biases 0.2f 0.1f 200
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