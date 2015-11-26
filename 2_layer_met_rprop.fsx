//Record: 98.9%. With max norm constraints.
type activationType = Relu | KSparse
let activation_type = Relu

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
let batch_size = 60000
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

// Finds the index of the max element of each column.
let rowReducer = new maxRowReduceModule<float32>()

// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

// Relu activation
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a >= 0.0f then a else 0.0f @>

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

// Max norm regularization module. It renormalizes the matrix to c if the euclidean norm is greater than c.
let maxNormModule784 = new maxNormRegularizationModule(784)
let maxNormModule1024 = new maxNormRegularizationModule(1024)

//let weights = load_weights_mnist @"C:\!NN\k=150,epochs=800,784x1024 weights (supervised, max_norm)" 784//load_weights_mnist @"C:\!NN\k=30,epochs=500,784x1024 weights (fine tuned)" 784//createRandomUniformMatrix 784 1024 1e-3f
//let weights2 = load_weights_mnist @"C:\!NN\k=150,epochs=800,1024x1024 weights_layer2 (supervised, max_norm)" 1024//load_weights_mnist @"C:\!NN\k=30,epochs=500,1024x1024 weights_layer2 (fine tuned)" 1024//createRandomUniformMatrix 1024 1024 1e-3f
//let weights3 = load_weights_mnist @"C:\!NN\k=150,epochs=800,1024x10 weights_layer3 (supervised, max_norm)" 1024//load_weights_mnist @"C:\!NN\k=30,epochs=500,1024x10 weights_layer3" 1024//createRandomUniformMatrix 1024 10 1e-2f
//let weights = load_weights_mnist @"C:\!NN\k=30,epochs=500,784x1024 weights (fine tuned)" 784//createRandomUniformMatrix 784 1024 1e-3f
//let weights2 = load_weights_mnist @"C:\!NN\k=30,epochs=500,1024x1024 weights_layer2 (fine tuned)" 1024//createRandomUniformMatrix 1024 1024 1e-3f
//let weights3 = load_weights_mnist @"C:\!NN\k=30,epochs=500,1024x10 weights_layer3" 1024//createRandomUniformMatrix 1024 10 1e-2f
let weights = createRandomUniformMatrix 784 1024 1e-3f
let weights2 = createRandomUniformMatrix 1024 1024 1e-3f
let weights3 = createRandomUniformMatrix 1024 10 1e-2f

let grad1_prev = createEmptyMatrixLike weights
let grad2_prev = createEmptyMatrixLike weights2
let grad3_prev = createEmptyMatrixLike weights3

setModule.Apply(0.0f,grad1_prev,grad1_prev)
setModule.Apply(0.0f,grad2_prev,grad2_prev)
setModule.Apply(0.0f,grad3_prev,grad3_prev)

let grad1_cur = createEmptyMatrixLike weights
let grad2_cur = createEmptyMatrixLike weights2
let grad3_cur = createEmptyMatrixLike weights3

let delta1 = createEmptyMatrixLike weights
let delta2 = createEmptyMatrixLike weights2
let delta3 = createEmptyMatrixLike weights3

setModule.Apply(0.0001f,delta1,delta1)
setModule.Apply(0.0001f,delta2,delta2)
setModule.Apply(0.0001f,delta3,delta3)

let delta_max = 1.0f
let delta_min = 1e-6f
let delta_plus = 1.2f
let delta_minus = 0.5f

let rpropMinusModule = 
    new DeviceTrinaryTransformModule<float32> 
        <@ fun delta_prev grad_prev grad_cur ->
        if grad_prev*grad_cur > 0.0f then
            min (delta_prev*delta_plus) delta_max
        else if grad_prev*grad_cur < 0.0f then
            max (delta_prev*delta_minus) delta_min
        else delta_prev @>

let rpropMinusWeightsAddModule =
    new DeviceTrinaryTransformModule<float32> 
        <@ fun weight_prev grad delta ->
        let sign_grad = 
            if grad > 0.0f then 1.0f
            else if grad < 0.0f then -1.0f
            else 0.0f
        weight_prev - sign_grad * delta @>

let manhattanModule =
    new DeviceBinaryCoefTransformModule<float32> 
        <@ fun _ weight_prev delta grad ->
        let sign_grad = 
            if grad > 0.0f then 1.0f
            else if grad < 0.0f then -1.0f
            else 0.0f
        weight_prev - sign_grad * delta @>

let batch,l = training_batches.[0]
    
let inv_batch_size = 1.0f / float32 batch.num_cols

// Preallocated memory. The modules also get compiled the first time they are run.
let z1 = sgemm T nT 1.0f weights batch
let z2 = sgemm T nT 1.0f weights2 z1
let z3 = sgemm T nT 1.0f weights3 z2

let squared_cost_error1 = binaryErrorModule.Apply(l, z3)
let squared_cost_error2 = sgemm nT nT 1.0f weights3 squared_cost_error1
let squared_cost_error3 = sgemm nT nT 1.0f weights2 squared_cost_error2
    
// Memory for the cost function of the test set.
let z1_c = sgemm T nT 1.0f weights dtest_data
let z2_c = sgemm T nT 1.0f weights2 z1_c
let z3_c = sgemm T nT 1.0f weights3 z2_c

type updateRule = Manhattan | SGD | Resilient

let sparsenet_fine_tune weights weights2 weights3 learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data start_k norm_constraint_l1 norm_constraint_l2 norm_constraint_l3 (update_rule: updateRule) =
    
    let crossEntropyError (batch: dM) l start_k =
        let alpha = - 1.0f / float32 batch.num_cols

        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        match activation_type with
            Relu -> reluActivationModule.Apply(z1,z1) |> ignore
            | KSparse -> sparsePiecewiseLinearActivation.Apply(z1, start_k, z1) |> ignore
        let z2 = sgemm2 T nT 1.0f weights2 z1 0.0f z2_c
        match activation_type with
            Relu -> reluActivationModule.Apply(z2,z2) |> ignore
            | KSparse -> sparsePiecewiseLinearActivation.Apply(z2, start_k, z2) |> ignore
        let z3 = sgemm2 T nT 1.0f weights3 z2 0.0f z3_c
        logisticActivationModule.Apply(z3, z3) |> ignore
        
        let cross_entropy_cost = alpha * crossEntropyCostModule.Apply(l, z3)
        cross_entropy_cost

    let gradient (batch: dM) l start_k grad1_cur grad2_cur grad3_cur grad1_prev grad2_prev grad3_prev delta1 delta2 delta3 = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        match activation_type with
            Relu -> reluActivationModule.Apply(z1,z1) |> ignore
            | KSparse -> sparsePiecewiseLinearActivation.Apply(z1, start_k, z1) |> ignore
        let z2 = sgemm2 T nT 1.0f weights2 z1 0.0f z2
        match activation_type with
            Relu -> reluActivationModule.Apply(z2,z2) |> ignore
            | KSparse -> sparsePiecewiseLinearActivation.Apply(z2, start_k, z2) |> ignore
        let z3 = sgemm2 T nT 1.0f weights3 z2 0.0f z3
        logisticActivationModule.Apply(z3, z3) |> ignore

        let squared_cost_error1 = binaryErrorModule.Apply(l, z3, squared_cost_error1)
        //sgemm2 nT T (-inv_batch_size*learning_rate) z2 squared_cost_error1 1.0f weights3 |> ignore
        sgemm2 nT T (inv_batch_size) z2 squared_cost_error1 0.0f grad3_cur |> ignore
        rpropMinusModule.Apply(delta3, grad3_prev, grad3_cur, delta3) |> ignore


        let squared_cost_error2 = sgemm2 nT nT 1.0f weights3 squared_cost_error1 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, z2, squared_cost_error2) |> ignore
        //sgemm2 nT T (-inv_batch_size*learning_rate) z1 squared_cost_error2 1.0f weights2 |> ignore
        sgemm2 nT T (inv_batch_size) z1 squared_cost_error2 0.0f grad2_cur |> ignore
        rpropMinusModule.Apply(delta2, grad2_prev, grad2_cur, delta2) |> ignore

        let squared_cost_error3 = sgemm2 nT nT 1.0f weights2 squared_cost_error2 0.0f squared_cost_error3
        binarySparseErrorModule.Apply(squared_cost_error3, z1, squared_cost_error3) |> ignore
        //sgemm2 nT T (-inv_batch_size*learning_rate) batch squared_cost_error3 1.0f weights |> ignore
        sgemm2 nT T (inv_batch_size) batch squared_cost_error3 0.0f grad1_cur |> ignore
        rpropMinusModule.Apply(delta1, grad1_prev, grad1_cur, delta1) |> ignore


        match update_rule with
            | Manhattan ->
                manhattanModule.Apply(1.0f, weights3, learning_rate, grad3_cur, weights3) |> ignore
                manhattanModule.Apply(1.0f, weights2, learning_rate, grad2_cur, weights2) |> ignore
                manhattanModule.Apply(1.0f, weights, learning_rate, grad1_cur, weights) |> ignore
            | Resilient ->
                rpropMinusWeightsAddModule.Apply(weights3,grad3_cur,delta3,weights3) |> ignore
                rpropMinusWeightsAddModule.Apply(weights2,grad2_cur,delta2,weights2) |> ignore
                rpropMinusWeightsAddModule.Apply(weights,grad1_cur,delta1,weights) |> ignore
            | SGD ->
                sgeam2 nT nT (-learning_rate) grad3_cur 1.0f weights3 weights3 |> ignore
                sgeam2 nT nT (-learning_rate) grad2_cur 1.0f weights2 weights2 |> ignore
                sgeam2 nT nT (-learning_rate) grad1_cur 1.0f weights weights |> ignore


    printfn "Cross entropy cost (before optimization) is %f" (crossEntropyError dtest_data dtest_label start_k)

    let mutable c = 0
    for epoch=1 to num_epochs do
        let current_k = start_k//max min_k (start_k - (epoch-1)*step_k)
        for batch,l in training_batches do
            if c % 2 = 0 then gradient batch l current_k grad1_cur grad2_cur grad3_cur grad1_prev grad2_prev grad3_prev delta1 delta2 delta3
            else gradient batch l current_k grad1_prev grad2_prev grad3_prev grad1_cur grad2_cur grad3_cur delta1 delta2 delta3
            //maxNormModule784.Apply(weights, weights, norm_constraint_l1) |> ignore
            //maxNormModule1024.Apply(weights2, weights2, norm_constraint_l2) |> ignore
            //maxNormModule1024.Apply(weights3, weights3, norm_constraint_l3) |> ignore
            c <- c + 1

        printfn "Cross entropy cost after epoch %i is %f" epoch (crossEntropyError dtest_data dtest_label current_k)

let test_time (batch: dM) weights weights2 weights3 start_k =
    let alpha = - 1.0f / float32 batch.num_cols

    let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
    match activation_type with
        Relu -> reluActivationModule.Apply(z1,z1) |> ignore
        | KSparse -> sparsePiecewiseLinearActivation.Apply(z1, start_k, z1) |> ignore
    let z2 = sgemm2 T nT 1.0f weights2 z1 0.0f z2_c
    match activation_type with
        Relu -> reluActivationModule.Apply(z2,z2) |> ignore
        | KSparse -> sparsePiecewiseLinearActivation.Apply(z2, start_k, z2) |> ignore
    let z3 = sgemm2 T nT 1.0f weights3 z2 0.0f z3_c

    let max_pred = rowReducer.Apply(z3)
    let max_labels = rowReducer.Apply(dtest_label)

    let pr,l = max_pred.Gather(), max_labels.Gather()

    let mutable c = 0
    for i=0 to pr.Length-1 do
        if pr.[i] = l.[i] then c <- c + 1
    printfn "The accuracy is %i/%i" c pr.Length

let save_bitmap (weights: dM) =
    let num_rows_and_cols = sqrt(float weights.num_rows) |> int
    let bitmap = make_bitmap_from_imageset weights num_rows_and_cols num_rows_and_cols 40 25
    bitmap.Save(@"C:\!NN\supervised_1a.bmp")

sparsenet_fine_tune weights weights2 weights3 0.04f 10 training_batches dtest_data 100 5.0f 5.0f 5.0f Resilient
test_time dtest_data weights weights2 weights3 100

//save_weights @"C:\!NN\k=250,epochs=1600,784x1024 weights (supervised, max_norm, 0.01f)" weights
//save_weights @"C:\!NN\k=250,epochs=1600,1024x1024 weights_layer2 (supervised, max_norm, 0.01f)" weights2
//save_weights @"C:\!NN\k=250,epochs=1600,1024x10 weights_layer3 (supervised, max_norm, 0.01f)" weights3

//save_bitmap weights

