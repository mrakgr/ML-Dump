// I am not particularly impressed by L1.
// I would guess that making save points every 50 epochs would be a better idea
// for combatting overfitting.

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

// The absolute sum of all the elements
let absoluteSumModule = new DeviceUnaryMapReduceModule <@ fun x -> abs(x) @>

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>
// For computing the error in the final layer with the sparse activation function.
// let trinarySparseErrorModule = new DeviceTrinaryTransformModule<float32> <@ fun y a c -> if c <> 0.0f then a-y else 0.0f @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

let hidden_layer_width = 1024
// The sparse activation function module inspired by the k-sparse autoencoder.
// http://arxiv.org/abs/1312.5663
let sparseActivationModule = new sparsePiecewiseLinearActivationModule(hidden_layer_width)

let sparse_autoencoder learning_rate lambda num_epochs start_k min_k =
    let batch,_ = training_batches.[0]
    let weights = createRandomUniformMatrix batch.num_rows hidden_layer_width 1e-5f

    let inv_batch_size = 1.0f / float32 batch.num_cols

    // The binary transform for the L1 regularization. I need to put it here because
    // otherwise I would not be able to pass inv_batch_size*lamda as a constant.
    let l1RegularizationModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> a + inv_batch_size*lambda*(if b >= 0.0f then 1.0f else -1.0f)@>

    // Preallocated memory.The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 
    let a1 = sparseActivationModule.Apply(z1,100)
    let z2 = sgemm nT nT 1.0f weights a1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let grad_second_layer = sgemm nT T inv_batch_size squared_cost_error z1
    let squared_cost_error2 = sgemm T nT 1.0f squared_cost_error weights
    binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
    let grad_first_layer = sgemm nT nT inv_batch_size batch squared_cost_error2 

    // Memory for the cost function of the test set.
    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c

    let costSquaredError batch weights k =
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c 
        let a1 = sparseActivationModule.Apply(z1,k, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) / float32 batch.num_cols + inv_batch_size*lambda*absoluteSumModule.Apply(weights)

    let gradient batch weights k = 
        let inv_batch_size = 1.0f / float32 batch.num_cols

        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = sparseActivationModule.Apply(z1,k,a1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        let grad_second_layer = sgemm2 nT T inv_batch_size squared_cost_error a1 0.0f grad_second_layer
        let squared_cost_error2 = sgemm2 T nT 1.0f squared_cost_error weights 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
        let grad_first_layer = sgemm2 nT nT inv_batch_size batch squared_cost_error2 0.0f grad_first_layer

        /// Add the weight gradients together in the first layer
        sgeam2 nT nT 1.0f grad_second_layer 1.0f grad_first_layer grad_first_layer |> ignore
        /// L1 penalty added to the weights
        l1RegularizationModule.Apply(grad_first_layer, weights, grad_first_layer)
                
    // In standard deviations.
    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data weights start_k)

    for epoch=1 to num_epochs do
        let current_k = max min_k (start_k-epoch+1)
        for batch,_ in training_batches do
            let grad = gradient batch weights current_k
            // Add them to the weights.
            sgeam2 nT nT 1.0f weights (-learning_rate) grad weights |> ignore

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data weights current_k)
        printfn "current_k is %i" current_k
    weights
        
#time
let weights = sparse_autoencoder 0.01f 5e-5f 200 100 25
#time

let batch,_ = training_batches.[0]

let z1 = sgemm T nT 1.0f weights batch 
let a1 = sparseActivationModule.Apply(z1,50, z1)
let z2 = sgemm nT nT 1.0f weights a1 

let bitmap = make_bitmap_from_imageset weights 28 28 40 25
bitmap.Save(@"C:\!NN\L1-k-sparse_5.bmp")


