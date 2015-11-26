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

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                let b_max = min 0.999999f b
                                let b_min = max 0.000001f b
                                a*(log b_min) + (1.0f-a)*log (1.0f - b_max)@>

let sigmoid_act = cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID

let batch,label = training_batches.[0]
let filters = createRandomUniform4DMatrix 16 1 5 5 1e-2f
let convLayer1 = new ConvLayer(batch,filters)
let l1 = convLayer1.convolutionForward(1.0f,batch,0.0f)

let filters2 = createRandomUniform4DMatrix 10 16 24 24 1e-4f
let convLayer2 = new ConvLayer(l1,filters2)

//let batch,label = training_batches.[0]
//let filters = createRandomUniform4DMatrix 16 1 5 5 1e-2f
let convLayer1test = new ConvLayer(batch,filters)
let l1 = convLayer1.convolutionForward(1.0f,batch,0.0f)

let filters2 = createRandomUniform4DMatrix 10 16 24 24 1e-4f
let convLayer2 = new ConvLayer(l1,filters2)


let inv_batch_size = 1.0f / float32 batch.num_feature_maps
let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
convLayer1.ActivationForward(1.0f,l1,0.0f,l1,relu_act)

let desc1 = convLayer2.getSourceData
convLayer2.convolutionBackwardData(1.f, l1, 0.f,desc1,deconvolutional_matrix)
inv_batch_size * squaredCostModule.Apply(batch.dArray.Length,batch.dArray.Ptr,deconvolutional_matrix.dArray.Ptr)

let conv_2_layer num_epochs learning_rate momentum_rate =
    let crossEntropyCost (batch: d4M) label =
        let alpha = -1.0f / float32 batch.num_feature_maps
        let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
        convLayer1.ActivationForward(1.0f,l1,0.0f,l1,relu_act)
        
        let l2 = convLayer2.convolutionForward(1.f, l1, 0.f)
        convLayer2.ActivationForward(1.0f,l2,0.0f,l2,sigmoid_act)
        alpha * crossEntropyCostModule.Apply(l2.dArray.Length,label.dArray.Ptr,l2.dArray.Ptr)

    let gradient batch label =
        // Here I implement Nesterov's Accelerated Gradient. I add momentum to the filters.
        // The gradient matrices can be reused neatly for this.
        // For some reason it works horribly for logistic regression.
        let grad2 = convLayer2.getGradientMatrix
        let grad1 = convLayer1.getGradientMatrix

        saxpy2 (momentum_rate) grad2 filters2
        saxpy2 (momentum_rate) grad1 filters

        let inv_batch_size = 1.0f / float32 batch.num_feature_maps
        let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
        convLayer1.ActivationForward(1.0f,l1,0.0f,l1,relu_act)
        let l2 = convLayer2.convolutionForward(1.f, l1, 0.f)
        convLayer2.ActivationForward(1.0f,l2,0.0f,l2,sigmoid_act)
        let er2 = convLayer2.lastLayerError(l2,label)
        let desc1, er1 = convLayer1.getErrorData
        convLayer2.convolutionBackwardData(1.0f,er2,0.0f,desc1,er1)
        convLayer1.activationBackward(1.0f,l1,er1,0.0f,er1,sigmoid_act)

        saxpy2 (-momentum_rate) grad2 filters2
        saxpy2 (-momentum_rate) grad1 filters

        let grad2 = convLayer2.convolutionBackwardFilter(-learning_rate*inv_batch_size,l1,er2,momentum_rate)
        let grad1 = convLayer1.convolutionBackwardFilter(-learning_rate*inv_batch_size,batch,er1,momentum_rate)

        saxpy2 1.0f grad2 filters2
        saxpy2 1.0f grad1 filters

    let calculate_validation_error() =
        let mutable c = 0.0f
        for batch,l in testing_batches do
            c <- c + (crossEntropyCost batch l)
        c / float32 testing_batches.Length

    printfn "Cross entropy error of the logistic regression layer before optimization is %f" (calculate_validation_error())

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            gradient batch l

        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (calculate_validation_error())

let test_time() =
    let calculate_accuracy batch (label: d4M) =
        let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
        convLayer1.ActivationForward(1.0f,l1,0.0f,l1,relu_act)
        let l2 = convLayer2.convolutionForward(1.f, l1, 0.f)
        //convLayer2.ActivationForward(1.0f,l2,0.0f,l2,sigmoid_act)

        let rowReducer = new maxRowReduceModule<float32>()

        let max_pred = rowReducer.Apply(l2)
        let max_labels = rowReducer.Apply(label)

        let pr,l = max_pred.Gather(), max_labels.Gather()

        let mutable c = 0
        for i=0 to pr.Length-1 do
            if pr.[i] = l.[i] then c <- c + 1
        c
    let mutable c = 0
    for batch,l in testing_batches do
        c <- c + (calculate_accuracy batch l)
    printfn "The accuracy is %i/%i" c 10000
    
conv_2_layer 40 0.000001f 0.999f
fillRandomUniformMatrix4D 1e-5f filters2
test_time()

