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
let batch_size = 125
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

type sparseWTAConvolutionalActivation(target, sample: d4M, k) =
    inherit GPUModule(target)
    
    let grid_size = 384
    let block_size = 32

    let n,c,h,w = sample.num_feature_maps, sample.num_channels, sample.num_rows, sample.num_cols

    let num_cols = c*n
    let num_rows = h*w

    let poolingLayer = new PoolingLayer(sample,max_pooling,h,w,0,0,1,1,false)

    let sparseWTAActivation = new sparseWTAActivationModule(GPUModuleTarget.Worker(worker),n,c,k,true) 

    new (sample, k) = new sparseWTAConvolutionalActivation(GPUModuleTarget.Worker(worker), sample, k)

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (x:deviceptr<float32>) (y:deviceptr<float32>) (z:deviceptr<float32>) =
               
        let num_vars = divup num_rows 32

        // Point block_start to where the column starts in the array.
        let mutable col = blockIdx.x
        
        while col < num_cols do
            let t = y.[col]
            __unroll()
            for i=0 to num_vars-1 do
                // idx is the absolute index in the array
                let row = threadIdx.x + i*32
                let idx = row + col * num_rows
                if row < num_rows then
                    if t = 0.0f || x.[idx] < t then z.[idx] <- 0.0f
                    else z.[idx] <- x.[idx]

            col <- col + gridDim.x

    member this.Apply(input: d4M, output: d4M) =
        if n <> input.num_feature_maps then failwith "n <> input.num_feature_maps in sparseWTAConvolutionalActivation"
        if c <> input.num_channels then failwith "n <> input.num_channels in sparseWTAConvolutionalActivation"
        if h <> input.num_rows then failwith "n <> input.num_rows in sparseWTAConvolutionalActivation"
        if w <> input.num_cols then failwith "n <> input.num_cols in sparseWTAConvolutionalActivation"

        if n <> output.num_feature_maps then failwith "n <> output.num_feature_maps in sparseWTAConvolutionalActivation"
        if c <> output.num_channels then failwith "n <> output.num_channels in sparseWTAConvolutionalActivation"
        if h <> output.num_rows then failwith "n <> output.num_rows in sparseWTAConvolutionalActivation"
        if w <> output.num_cols then failwith "n <> output.num_cols in sparseWTAConvolutionalActivation"

        if input.dArray.Length <> sample.dArray.Length then failwith "input.dArray.Length <> sample.dArray.Length in sparseWTAConvolutionalActivation"
        if output.dArray.Length <> sample.dArray.Length then failwith "output.dArray.Length <> sample.dArray.Length in sparseWTAConvolutionalActivation"

        let l1 = poolingLayer.poolingForward(1.0f,input,0.0f)
        let dl1: dM = {num_rows=c; num_cols=n; dArray=l1.dArray}
        sparseWTAActivation.ApplyTranspose(dl1,dl1) |> ignore

        let lp = LaunchParam(min grid_size input.num_cols, block_size)
        this.GPULaunch <@ this.Kernel @> lp input.dArray.Ptr l1.dArray.Ptr output.dArray.Ptr
        output

// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

// Relu activation
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a >= 0.0f then a else 0.0f @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

let batch,label = training_batches.[0]
let filters = createRandomUniform4DMatrix 128 1 5 5 1e-2f
let convLayer1 = new ConvLayer(batch,filters)
let l1 = convLayer1.convolutionForward(1.0f,batch,0.0f)

let deconvolutional_matrix = createEmpty4DMatrixLike batch
let convLayer2 = new ConvLayer(deconvolutional_matrix, filters)
let r = l1.num_feature_maps*l1.num_channels
let l = l1.num_cols*l1.num_rows

let sparseWTAActivation = new sparseWTAConvolutionalActivation(l1,6)
//let sparseWTAActivation = new sparseWTAActivationModule(GPUModuleTarget.Worker(worker),l,r,1,false) 
//let sparseWTAActivation = new sparseWTAActivationModule(GPUModuleTarget.Worker(worker),l1.num_feature_maps,l1.num_channels,6,true) 
//let dl1: dM = {num_rows=l; num_cols=r; dArray=l1.dArray}
//let dl1: dM = {num_rows=l1.num_channels; num_cols=l1.num_feature_maps; dArray=l1.dArray}

let conv_autoencoder num_epochs learning_rate momentum_rate =
    let squaredCost (batch: d4M) =
        let inv_batch_size = 1.0f / float32 batch.num_feature_maps
        let l1 = convLayer1.convolutionForward(1.f, batch, 0.f)
        //convLayer1.ActivationForward(1.0f,l1,0.0f,l1,relu_act)
        sparseWTAActivation.Apply(l1,l1) |> ignore
        //sparseWTAActivation.ApplyTranspose(dl1,dl1) |> ignore

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
        sparseWTAActivation.Apply(l1,l1) |> ignore
        //sparseWTAActivation.ApplyTranspose(dl1,dl1) |> ignore

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
            c <- c + (squaredCost batch)
        c / float32 testing_batches.Length

    printfn "Squared cost error before optimization is %f" (calculate_validation_error())

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            gradient batch

        printfn "Squared cost error after epoch %i is %f" epoch (calculate_validation_error())

conv_autoencoder 5 0.0002f 0.99f
//fillRandomUniformMatrix4D 1e-3f filters

let save_bitmap (weights: d4M) =
    let num_rows_and_cols = sqrt(float weights.num_rows * float weights.num_cols) |> int
    let weights_dm = {num_rows=num_rows_and_cols*num_rows_and_cols;num_cols=weights.num_feature_maps;dArray=weights.dArray}:dM
    let bitmap = make_bitmap_from_imageset weights_dm num_rows_and_cols num_rows_and_cols 16 8
    bitmap.Save(@"C:\!NN\wta_sparse_convolutional_1f.bmp")

save_bitmap filters