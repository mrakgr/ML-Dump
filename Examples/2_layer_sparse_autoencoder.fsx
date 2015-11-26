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
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

// The sparse activation.
let sparsePiecewiseLinearActivation = new sparsePiecewiseLinearActivationModule(1024,50)

/// Logistic(x)
//let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>
let logisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> 
        if x <= 0.0f then 0.0f
        else if x >= 1.0f then 1.0f
        else x @>

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                let b_max = min 0.999999f b
                                let b_min = max 0.000001f b
                                a*(log b_min) + (1.0f-a)*log (1.0f - b_max)@>

let logistic_regression weights learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data = 
    let batch,_ = training_batches.[0]
    let inv_batch_size = 1.0f / float32 batch.num_cols

    let z1 = sgemm T nT 1.0f weights batch

    let z1_c = sgemm T nT 1.0f weights dtest_data 

    let costLogRegression batch weights labels =
        let alpha = -1.0f/float32 batch.num_cols

        ///logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        let a1 = logisticActivationModule.Apply(z1, z1)

        let cross_entropy_cost = alpha * crossEntropyCostModule.Apply(labels, a1)
        
        cross_entropy_cost

    let gradient batch weights labels =
        /// logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = logisticActivationModule.Apply(z1,z1)
        /// -(labels-a1)
        let cross_entropy_error = binaryErrorModule.Apply(labels,a1,a1)
        /// Add directly to the weights.
        sgemm2 nT T (-learning_rate*inv_batch_size) batch cross_entropy_error 1.0f weights |> ignore

    printfn "Cross entropy error of the logistic regression layer before optimization is %f" (costLogRegression dtest_data weights dtest_label)

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            gradient batch weights l

        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (costLogRegression dtest_data weights dtest_label)

type updateRule = Manhattan | SGD

let manhattanModule =
    new DeviceBinaryCoefTransformModule<float32> 
        <@ fun _ weight_prev delta grad ->
        let sign_grad = 
            if grad > 0.0f then 1.0f
            else if grad < 0.0f then -1.0f
            else 0.0f
        weight_prev - sign_grad * delta @>

let sparse_autoencoder_fine_tune weights weights2 learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data start_k (update_rule: updateRule) =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    let grad1 = createEmptyMatrixLike weights
    let grad2 = createEmptyMatrixLike weights2

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch
    let z2 = sgemm T nT 1.0f weights2 z1
    let z3 = sgemm nT nT 1.0f weights2 z2
    let z4 = sgemm nT nT 1.0f weights z3

    let squared_cost_error1 = binaryErrorModule.Apply(batch, z4)
    let squared_cost_error2 = sgemm T nT 1.0f weights squared_cost_error1
    let squared_cost_error3 = sgemm T nT 1.0f weights2 squared_cost_error2
    let squared_cost_error4 = sgemm nT nT 1.0f weights2 squared_cost_error3
    
    // Memory for the cost function of the test set.
    // Here it is a bit long. I should really do this all using a single array...
    // Hopefully I won't forget that I now have a limit of 1024 for the hidden layer size.
    let z1_c = sgemm T nT 1.0f weights dtest_data
    let z2_c = sgemm T nT 1.0f weights2 z1_c
    let z3_c = sgemm nT nT 1.0f weights2 z2_c
    let z4_c = sgemm nT nT 1.0f weights z3_c
    
    let costSquaredError batch start_k =
        let inv_batch_size = 1.0f / float32 batch.num_cols

        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        sparsePiecewiseLinearActivation.Apply(z1, start_k, z1) |> ignore
        let z2 = sgemm2 T nT 1.0f weights2 z1 0.0f z2_c
        sparsePiecewiseLinearActivation.Apply(z2, start_k, z2) |> ignore
        let z3 = sgemm2 nT nT 1.0f weights2 z2 0.0f z3_c
        sparsePiecewiseLinearActivation.Apply(z3, start_k, z3) |> ignore
        let z4 = sgemm2 nT nT 1.0f weights z3 0.0f z4_c

        squaredCostModule.Apply(batch.dArray.Length, batch.dArray.Ptr, z4.dArray.Ptr) * inv_batch_size

    let gradient batch start_k = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        sparsePiecewiseLinearActivation.Apply(z1, start_k, z1) |> ignore
        let z2 = sgemm2 T nT 1.0f weights2 z1 0.0f z2
        sparsePiecewiseLinearActivation.Apply(z2, start_k, z2) |> ignore
        let z3 = sgemm2 nT nT 1.0f weights2 z2 0.0f z3
        sparsePiecewiseLinearActivation.Apply(z3, start_k, z3) |> ignore
        let z4 = sgemm2 nT nT 1.0f weights z3 0.0f z4

        let squared_cost_error1 = binaryErrorModule.Apply(batch, z4, squared_cost_error1)
        
        let squared_cost_error2 = sgemm2 T nT 1.0f weights squared_cost_error1 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, z3, squared_cost_error2) |> ignore
        
        let squared_cost_error3 = sgemm2 T nT 1.0f weights2 squared_cost_error2 0.0f squared_cost_error3
        binarySparseErrorModule.Apply(squared_cost_error3, z2, squared_cost_error3) |> ignore
        
        let squared_cost_error4 = sgemm2 nT nT 1.0f weights2 squared_cost_error3 0.0f squared_cost_error4
        binarySparseErrorModule.Apply(squared_cost_error4, z1, squared_cost_error4) |> ignore

        match update_rule with
            | SGD -> 
                sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error1 z3 1.0f weights |> ignore
                sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error2 z2 1.0f weights2 |> ignore
                sgemm2 nT T (-inv_batch_size*learning_rate) z1 squared_cost_error3 1.0f weights2 |> ignore
                sgemm2 nT T (-inv_batch_size*learning_rate) batch squared_cost_error4 1.0f weights |> ignore
            | Manhattan -> 
                sgemm2 nT T (inv_batch_size) squared_cost_error1 z3 0.0f grad1 |> ignore
                sgemm2 nT T (inv_batch_size) squared_cost_error2 z2 0.0f grad2 |> ignore
                sgemm2 nT T (inv_batch_size) z1 squared_cost_error3 0.0f grad2 |> ignore
                sgemm2 nT T (inv_batch_size) batch squared_cost_error4 0.0f grad1 |> ignore

                manhattanModule.Apply(1.0f, weights2, learning_rate, grad2, weights2) |> ignore
                manhattanModule.Apply(1.0f, weights, learning_rate, grad1, weights) |> ignore


    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data start_k)

    for epoch=1 to num_epochs do
        let current_k = start_k//max min_k (start_k - (epoch-1)*step_k)
        for batch,_ in training_batches do
            gradient batch current_k

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data current_k)

let test_time training_batches_sparse dtest_data_sparse =
    let weights2 = createRandomUniformMatrix dtest_data_sparse.num_rows 10 1e-3f
    logistic_regression weights2 1.0f 500 training_batches_sparse dtest_data_sparse

    let rowReducer = new maxRowReduceModule<float32>()

    let predictions = sgemm T nT 1.0f weights2 dtest_data_sparse
    let max_pred = rowReducer.Apply(predictions)
    let max_labels = rowReducer.Apply(dtest_label)

    let pr,l = max_pred.Gather(), max_labels.Gather()

    let mutable c = 0
    for i=0 to pr.Length-1 do
        if pr.[i] = l.[i] then c <- c + 1
    printfn "The accuracy is %i/%i" c pr.Length

let save_bitmap weights =
    let num_rows_and_cols = sqrt(float weights.num_rows) |> int
    let bitmap = make_bitmap_from_imageset weights num_rows_and_cols num_rows_and_cols 25 10
    bitmap.Save(@"C:\!NN\fixed_stacked_relu_dropout_1a.bmp")

let weights = load_weights_mnist @"C:\!NN\k=30,epochs=500,784x1024 weights" 784
let weights2 = load_weights_mnist @"C:\!NN\k=30,epochs=500,1024x1024 weights_layer2" 1024

sparse_autoencoder_fine_tune weights weights2 0.005f 10 training_batches dtest_data 30 SGD

save_weights @"C:\!NN\k=30,epochs=500,784x1024 weights (fine tuned)" weights
save_weights @"C:\!NN\k=30,epochs=500,1024x1024 weights_layer2 (fine tuned)" weights2
