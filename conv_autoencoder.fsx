// A fully connected autoencoder using the convolutional functions.
// This is merely a test run to see if deconvolution works.
// It is really quite slow.
#load "convolution.fsx"
open Load_mnist.MnistLoad
open Utils.Utils
open Convolution

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
        let dtrain_data: d4M = 
                      {num_feature_maps = batch_size
                       num_channels = 1
                       num_rows = train.num_rows
                       num_cols = train.num_cols
                       dArray = worker.Malloc(train.float_data.[s1..s2])}

        if (dtrain_data.num_cols*dtrain_data.num_rows*dtrain_data.num_channels*dtrain_data.num_feature_maps <> dtrain_data.dArray.Length)
        then failwith "Invalid batch size (test)."

        let s1 = 10*i
        let s2 = 10*(i+batch_size)-1
        let dtrain_label: d4M =
                           {num_feature_maps = batch_size
                            num_channels = 10
                            num_rows = 1
                            num_cols = 1
                            dArray = worker.Malloc(train.float_labels.[s1..s2])}
        if (dtrain_label.num_cols*dtrain_label.num_rows*dtrain_label.num_channels*dtrain_label.num_feature_maps <> dtrain_label.dArray.Length)
        then failwith "Invalid batch size (label)."
        yield (dtrain_data, dtrain_label)|]

let testing_batches =
    [|
    for i in 0..batch_size..test.num_images-1 do
        let s1 = test.num_rows*test.num_cols*i
        let s2 = test.num_rows*test.num_cols*(i+batch_size)-1
        let dtest_data: d4M = 
                      {num_feature_maps = batch_size
                       num_channels = 1
                       num_rows = test.num_rows
                       num_cols = test.num_cols
                       dArray = worker.Malloc(test.float_data.[s1..s2])}

        if (dtest_data.num_cols*dtest_data.num_rows*dtest_data.num_channels*dtest_data.num_feature_maps <> dtest_data.dArray.Length)
        then failwith "Invalid batch size (test)."

        let s1 = 10*i
        let s2 = 10*(i+batch_size)-1
        let dtest_label: d4M =
                           {num_feature_maps = batch_size
                            num_channels = 10
                            num_rows = 1
                            num_cols = 1
                            dArray = worker.Malloc(test.float_labels.[s1..s2])}
        if (dtest_label.num_cols*dtest_label.num_rows*dtest_label.num_channels*dtest_label.num_feature_maps <> dtest_label.dArray.Length)
        then failwith "Invalid batch size (label)."
        yield (dtest_data, dtest_label)|]

// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

// Relu activation
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a >= 0.0f then a else 0.0f @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

let batch,label = training_batches.[0]
let filters = createRandomUniform4DMatrix 1024 1 28 28 1e-3f
let convLayer1 = new ConvLayer(batch,filters)

let deconvolutional_matrix = createEmpty4DMatrixLike batch
let convLayer2 = new ConvLayer(deconvolutional_matrix, filters)    

let conv_autoencoder num_epochs learning_rate momentum_rate =
    let crossEntropyCost (batch: d4M) =
        let inv_batch_size = 1.0f / float32 batch.num_feature_maps
        let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
        convLayer1.ActivationForward(1.0f,l1,0.0f,l1,relu_act)

        convLayer2.convolutionBackwardData(1.f, l1, 0.f,deconvolutional_matrix)
        inv_batch_size * squaredCostModule.Apply(batch.dArray.Length,batch.dArray.Ptr,deconvolutional_matrix.dArray.Ptr)

    let gradient batch =
        // Here I implement Nesterov's Accelerated Gradient. I add momentum to the filters.
        // The gradient matrices can be reused neatly for this.
        let grad2 = convLayer2.getGradientMatrix
        let grad1 = convLayer1.getGradientMatrix

        saxpy2 (momentum_rate) grad2 filters
        saxpy2 (momentum_rate) grad1 filters

        let inv_batch_size = 1.0f / float32 batch.num_feature_maps
        let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
        convLayer1.ActivationForward(1.0f,l1,0.0f,l1,relu_act)

        let desc1 = convLayer2.getSourceData
        convLayer2.convolutionBackwardData(1.f, l1, 0.f,deconvolutional_matrix)

        binaryErrorModule.Apply(batch.dArray.Length,batch.dArray.Ptr, deconvolutional_matrix.dArray.Ptr,deconvolutional_matrix.dArray.Ptr)
        let er1 = deconvolutional_matrix
        let er2 = convLayer2.convolutionForward(1.f,er1,0.f)
        binarySparseErrorModule.Apply(er2.dArray.Length,er2.dArray.Ptr, l1.dArray.Ptr, er2.dArray.Ptr) |> ignore

        saxpy2 (-momentum_rate) grad2 filters
        saxpy2 (-momentum_rate) grad1 filters

        let grad2 = convLayer2.convolutionBackwardFilter(-learning_rate*inv_batch_size,er1,l1,momentum_rate)
        let grad1 = convLayer1.convolutionBackwardFilter(-learning_rate*inv_batch_size,batch,er2,momentum_rate)

        saxpy2 1.0f grad2 filters
        saxpy2 1.0f grad1 filters

    let calculate_validation_error() =
        let mutable c = 0.0f
        for batch,l in testing_batches do
            c <- c + (crossEntropyCost batch)
        c / float32 testing_batches.Length

    printfn "Cross entropy error of the logistic regression layer before optimization is %f" (calculate_validation_error())

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            gradient batch

        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (calculate_validation_error())

conv_autoencoder 5 0.01f 0.99f
//fillRandomUniformMatrix4D 1e-3f filters


