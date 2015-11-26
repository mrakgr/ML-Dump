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

let dtest_data: d4M = 
                {num_feature_maps = test.num_images
                 num_channels = 1
                 num_rows = test.num_rows
                 num_cols = test.num_cols
                 dArray = worker.Malloc(test.float_data)}

let dtest_label: d4M =
                  {num_feature_maps = test.num_images
                   num_channels = 10
                   num_rows = 1
                   num_cols = 1
                   dArray = worker.Malloc(test.float_labels)}

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                let b_max = min 0.999999f b
                                let b_min = max 0.000001f b
                                a*(log b_min) + (1.0f-a)*log (1.0f - b_max)@>

let batch,label = training_batches.[0]
let filters = createRandomUniform4DMatrix 10 1 28 28 1e-3f
let convLayer1 = new ConvLayer(batch,filters)
let convLayer1test = new ConvLayer(dtest_data,filters)


let conv_logistic_regression num_epochs learning_rate momentum_rate =
    let crossEntropyCost (batch: d4M) label =
        let alpha = -1.0f / float32 batch.num_feature_maps
        let l1 = convLayer1test.convolutionForward(1.f, batch, 0.f)
        convLayer1test.ActivationForward(1.0f,l1,0.0f,l1,sigmoid_act)
        alpha * crossEntropyCostModule.Apply(l1.dArray.Length,label.dArray.Ptr,l1.dArray.Ptr)

    let gradient batch label =
        // Here I implement Nesterov's Accelerated Gradient. I add momentum to the filters.
        // The gradient matrices can be reused neatly for this.
        // For some reason it works horribly for logistic regression.
        // ...Figured it out in the other file. Because I should really be multipying the matrices by the learning rate before
        // adding them to the weights.
        let grad = convLayer1.getGradientMatrix
        //saxpy2 (momentum_rate) grad filters

        let inv_batch_size = 1.0f / float32 batch.num_feature_maps
        let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
        convLayer1.ActivationForward(1.0f,l1,0.0f,l1,sigmoid_act)
        let er = convLayer1.lastLayerError(l1,label)

        //saxpy2 (-momentum_rate) grad filters

        let grad = convLayer1.convolutionBackwardFilter(inv_batch_size,batch,er,momentum_rate)

        saxpy2 (-learning_rate) grad filters

    printfn "Cross entropy error of the logistic regression layer before optimization is %f" (crossEntropyCost dtest_data dtest_label)

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            gradient batch l

        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (crossEntropyCost dtest_data dtest_label)

conv_logistic_regression 30 0.1f 0.0f
fillRandomUniformMatrix4D 1e-3f filters 
