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
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c >= 0.0f then y else 0.0f @>

// Relu activation.
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a >= 0.0f then a else 0.0f @>
//let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> if x <= 0.0f then 0.0f else if x >= 1.0f then 1.0f else x @>

let relu_dropout_autoencoder_fine_tune weights1 weights2 learning_rate exp_decay1 exp_decay2 epsilon num_epochs (training_batches: (dM*dM) []) dtest_data grad_keepin_rate =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Applies a dropout according to an uniform random matrix b [0,1)
    //let dropoutModule = new DeviceBinaryCoefTransformModule<float32> <@ fun coef_a a coef_b b -> if b <= coef_b then a else 0.0f @>
    let dropout_weight_matrix = createEmptyMatrixLike weights1

    let grad1 = createEmptyMatrixLike weights1
    let grad2 = createEmptyMatrixLike weights2

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights1 batch
    let z2 = sgemm T nT 1.0f weights2 z1

    let squared_cost_error1 = binaryErrorModule.Apply(batch, z2)
    let squared_cost_error2 = sgemm T nT 1.0f weights1 squared_cost_error1
    
    // Memory for the cost function of the test set.
    // Here it is a bit long. I should really do this all using a single array...
    // Hopefully I won't forget that I now have a limit of 1024 for the hidden layer size.
    let z1_c = sgemm T nT 1.0f weights1 dtest_data
    let z2_c = sgemm T nT 1.0f weights2 z1_c
    //let z3_c = sgemm nT nT 1.0f weights3 z2_c
    //let z4_c = sgemm nT nT 1.0f weights4 z3_c
    
    let costSquaredError batch =
        let inv_batch_size = 1.0f / float32 batch.num_cols

        let z1 = sgemm2 T nT 1.0f weights1 batch 0.0f z1_c
        reluActivationModule.Apply(z1, z1) |> ignore
        let z2 = sgemm2 T nT 1.0f weights2 z1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) * inv_batch_size
        
    let gradient batch i = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights1 batch 0.0f z1
        reluActivationModule.Apply(z1, z1) |> ignore
        let z2 = sgemm2 T nT 1.0f weights2 z1 0.0f z2

        let squared_cost_error1 = binaryErrorModule.Apply(batch, z2, squared_cost_error1)
        sgemm2 nT T (-inv_batch_size * learning_rate) z1 squared_cost_error1 0.0f grad2 |> ignore
        //dropoutModule.Apply(1.0f,grad2,grad_keepin_rate,dropout_weight_matrix, grad2) |> ignore
        sgeam2 nT nT 1.0f grad2 1.0f weights2 weights2 |> ignore

        let squared_cost_error2 = sgemm2 T nT 1.0f weights1 squared_cost_error1 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, z1, squared_cost_error2) |> ignore
        sgemm2 nT T (-inv_batch_size * learning_rate) batch squared_cost_error2 0.0f grad1 |> ignore
        //dropoutModule.Apply(1.0f,grad1,grad_keepin_rate,dropout_weight_matrix, grad1) |> ignore
        sgeam2 nT nT 1.0f grad1 1.0f weights1 weights1 |> ignore

    (*
    let m1 = setModule.Apply(0.0f,weights)
    let m2 = setModule.Apply(0.0f,weights2)

    let v1 = setModule.Apply(0.0f,weights)
    let v2 = setModule.Apply(0.0f,weights2)

    let mutable exp_decay1_pow = 1.0f
    let mutable exp_decay2_pow = 1.0f


    // The Adam optimizer.
    // http://arxiv.org/abs/1412.6980
    let mModule = new DeviceBinaryTransformModule<float32> <@ fun x y -> exp_decay1*x + (1.0f-exp_decay1)*y @>
    let vModule = new DeviceBinaryTransformModule<float32> <@ fun x y -> exp_decay2*x + (1.0f-exp_decay2)*y*y @>
    let adamModule = 
        new DeviceTrinaryCoefTransformModule<float32> 
            <@ fun coef_x x coef_y y coef_z z -> x + coef_y*y/(sqrt(z)+epsilon)@>
    *)

    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data)

    for epoch=1 to num_epochs do
        fillRandomUniformMatrix 1.0f dropout_weight_matrix
        for batch,_ in training_batches do
            gradient batch epoch
            (*
            exp_decay1_pow <- exp_decay1_pow * exp_decay1
            exp_decay2_pow <- exp_decay2_pow * exp_decay2

            mModule.Apply(m1,grad1,m1) |> ignore
            mModule.Apply(m2,grad2,m2) |> ignore

            vModule.Apply(v1,grad1,v1) |> ignore
            vModule.Apply(v2,grad2,v2) |> ignore

            let learning_rate_t = -inv_batch_size * learning_rate * (sqrt (1.0f - exp_decay2_pow)) / (1.0f - exp_decay1_pow)
            adamModule.Apply(1.0f,weights1,learning_rate_t,m1,1.0f,v1, weights1) |> ignore
            adamModule.Apply(1.0f,weights2,learning_rate_t,m2,1.0f,v2, weights2) |> ignore
            *)

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data)


//let identity_matrix = Array.init (784*784) (fun i -> if i % 784 = i / 784 then 1.0f else 0.0f)
let weights1 = createRandomUniformMatrix 784 784 1e-3f
let weights2 = createRandomUniformMatrix 784 784 1e-3f

relu_dropout_autoencoder_fine_tune weights1 weights2 1e-2f 0.9f 0.999f 1e-8f 100 training_batches dtest_data 1.0f

