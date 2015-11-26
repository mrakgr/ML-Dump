// I'll give this a pass. Now that I've thought about it, I've realized that the addition of PrRelu would make the code
// complexity explode.

let keepin_rate = 0.2f

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

// For errors in the middle layers using sparse activations.
let parametricReluGradientModule = new DeviceTrinaryTransformModule<float32> <@ fun y c p -> if c >= 0.0f then y else y*p @>

// Relu activation.
let pReluActivationModule = 
    new DeviceBinaryTransformModule<float32> 
        <@ fun a p -> if a >= 0.0f then a else p*a @>

let batch,_ = training_batches.[0]
let weights = createRandomUniformMatrix 784 1024 1e-4f


let inv_batch_size = 1.0f / float32 batch.num_cols
let z1 = sgemm T nT 1.0f weights batch

let par_coef = createRandomUniformMatrix z1.num_rows z1.num_cols 1e-1f
let a1 = pReluActivationModule.Apply(z1, par_coef, z1)
let z2 = sgemm nT nT 1.0f weights a1

let learning_rate = 0.01f
let squared_cost_error = binaryErrorModule.Apply(batch, z2)
sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error a1 1.0f weights |> ignore
        
let squared_cost_error2 = sgemm T nT 1.0f squared_cost_error weights
parametricReluGradientModule.Apply(squared_cost_error2, a1, par_coef, squared_cost_error2) |> ignore
sgemm2 nT nT (-inv_batch_size*learning_rate) batch squared_cost_error2 1.0f weights |> ignore
sgemm2 nT nT (-inv_batch_size*learning_rate) batch squared_cost_error2 1.0f weights |> ignore
squared_cost_error2
par_coef
                
