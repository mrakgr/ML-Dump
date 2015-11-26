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
let training_batches =
    [|
    let step = 250
    for i in 0..step..train.num_images-1 do
        let s1 = train.num_rows*train.num_cols*i
        let s2 = train.num_rows*train.num_cols*(i+step)-1
        let dtrain_data: dM = 
                      {num_rows = train.num_rows*train.num_cols
                       num_cols = step
                       dArray = worker.Malloc(train.float_data.[s1..s2])}

        if (dtrain_data.num_cols*dtrain_data.num_rows <> dtrain_data.dArray.Length)
        then failwith "Invalid batch size (test)."

        let s1 = 10*i
        let s2 = 10*(i+step)-1
        let dtrain_label: dM =
                           {num_rows = 10
                            num_cols = step
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

let weights = createRandomUniformMatrix 784 100 1e-3f -0.5f
let bias_l1 = createRandomUniformMatrix 100 1 1e-3f -0.5f
let bias_l2 = createRandomUniformMatrix 784 1 1e-3f -0.5f

let batch1, _ = training_batches.[0]

let z1 = sgemm T nT 1.0f weights batch1 
addBias z1 bias_l1
let a1 = createEmptyMatrixLike z1
sigmoidActivationForward z1 a1

let z2 = sgemm nT nT 1.0f weights z1
addBias z2 bias_l2

let crossEntropyCostModule = new DeviceBinaryTransformModule<float32> 
                                    <@ fun y a -> y*(log a)+(1.0f-y)*(1.0f-log a)@>
let sumReduce: DeviceMemory<float32> -> float32 = 
    makeReduce <@ fun a b -> a + b @>

let squaredCost = 0.5f*sumReduce (crossEntropyCostModule.Apply(batch1, z2)).dArray

let distanceModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

/// -(y-a) = a-y
/// This is the error term in the last layer.
let cross_entropy_error = distanceModule.Apply(batch1, z2)

let grad

