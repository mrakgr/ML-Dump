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

//let weights = createRandomUniformMatrix 2 5 1e-0f -0.5f
//let bias_input = createRandomUniformMatrix 2 1 1e-0f -0.5f

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

    let weights = createRandomUniformMatrix 784 100 1e-5f

    let batch, _ = training_batches.[0]

    /// Prealocate memory.
    let z1 = sgemm T nT 1.0f weights batch 
    let z2 = sgemm nT nT 1.0f weights z1
    let squared_cost_error = distanceModule.Apply(batch, z2)
    let grad_second_layer = sgemm nT T inv_batch_size squared_cost_error z1
    let squared_cost_error2 = sgemm T nT 1.0f squared_cost_error weights
    let grad_first_layer = sgemm nT nT inv_batch_size batch squared_cost_error2 

    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c

    let costSquaredError batch weights =
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        let z2 = sgemm2 nT nT 1.0f weights z1 0.0f z2_c

        sumReduce.Apply(batch, z2) / float32 batch.num_cols


    let gradient batch weights = 

        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let z2 = sgemm2 nT nT 1.0f weights z1 0.0f z2

        /// -(y-a) = a-y
        /// This is the error term in the last layer...and in the following layers
        let squared_cost_error = distanceModule.Apply(batch, z2, squared_cost_error)

        let grad_second_layer = sgemm2 nT T inv_batch_size squared_cost_error z1 0.0f grad_second_layer

        /// Propagate the errors down.
        let squared_cost_error2 = sgemm2 T nT 1.0f squared_cost_error weights 0.0f squared_cost_error2

        let grad_first_layer = sgemm2 nT nT inv_batch_size batch squared_cost_error2 0.0f grad_first_layer

        /// Add the weight gradients together in the first layer
        sgeam2 nT nT 1.0f grad_second_layer 1.0f grad_first_layer grad_first_layer

    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data weights)

    for epoch=1 to num_epochs do
        for batch,_ in training_batches do
            let grad = gradient batch weights
            // Add them to the weights.
            sgeam2 nT nT 1.0f weights (-learning_rate) grad weights |> ignore

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data weights)
    weights
        
#time
let weights = linear_autoencoder 0.01f 200
#time

let z1 = sgemm T nT 1.0f weights dtest_data//(fst training_batches.[0])
let z2 = sgemm nT nT 1.0f weights z1

//let bitmap = make_bitmap_from_imageset reconstructed_images train.num_rows train.num_cols 10 15
//bitmap.Save(@"C:\!NN\reconstructions250_7.bmp")

let bitmap = make_bitmap_from_imageset z2 28 28 50 25
bitmap.Save(@"C:\!NN\reconstructions250_8.bmp")
