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

// Relu activation.
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a >= 0.0f then a else 0.0f @>

let relu_dropout_autoencoder weights learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data keepin_rate =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Applies a dropout according to an uniform random matrix b [0,1)
    let dropoutModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> if b <= keepin_rate then a else 0.0f @>

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 
    let dropout_matrix = createEmptyMatrixLike z1

    let z2 = sgemm nT nT 1.0f weights z1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let squared_cost_error2 = sgemm T nT 1.0f weights squared_cost_error
    
    // Memory for the cost function of the test set.
    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c
    
    let costSquaredError batch weights =
    
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT keepin_rate weights batch 0.0f z1_c 
        let a1 = reluActivationModule.Apply(z1, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) * inv_batch_size
        

    let gradient batch weights dropout_matrix = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = reluActivationModule.Apply(z1, z1)
        dropoutModule.Apply(a1, dropout_matrix, a1) |> ignore
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error a1 1.0f weights |> ignore
        let squared_cost_error2 = sgemm2 T nT 1.0f weights squared_cost_error 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) batch squared_cost_error2 1.0f weights |> ignore
                
    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data weights)

    for epoch=1 to num_epochs do
        fillRandomUniformMatrix 1.0f dropout_matrix
        for batch,_ in training_batches do
            gradient batch weights dropout_matrix

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data weights)


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

let feedforward_sparse_pass weights (training_batches: (dM*dM) []) dtest_data keepin_rate =
    let training_batches_sparse =
        [|for batch,l in training_batches do
            let z1 = sgemm T nT keepin_rate weights batch
            reluActivationModule.Apply(z1, z1) |> ignore
            yield z1, l|]

    let dtest_data_sparse = sgemm T nT keepin_rate weights dtest_data
    reluActivationModule.Apply(dtest_data_sparse, dtest_data_sparse) |> ignore
    training_batches_sparse, dtest_data_sparse

let relu_dropout_autoencoder_fine_tune weights weights2 learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data keepin_rate keepin_rate2 =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Applies a dropout according to an uniform random matrix b [0,1)
    let dropoutModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> if b <= keepin_rate then a else 0.0f @>
    let dropoutModule2 = new DeviceBinaryTransformModule<float32> <@ fun a b -> if b <= keepin_rate2 then a else 0.0f @>

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
    
    let costSquaredError batch =
        let inv_batch_size = 1.0f / float32 batch.num_cols

        let z1 = sgemm2 T nT keepin_rate weights batch 0.0f z1_c
        reluActivationModule.Apply(z1, z1) |> ignore
        let z2 = sgemm2 T nT keepin_rate2 weights2 z1 0.0f z2_c
        reluActivationModule.Apply(z2, z2) |> ignore
        let z3 = sgemm2 nT nT 1.0f weights2 z2 0.0f z3_c
        reluActivationModule.Apply(z3, z3) |> ignore
        let z4 = sgemm2 nT nT 1.0f weights z3 0.0f z4_c

        squaredCostModule.Apply(batch.dArray.Length, batch.dArray.Ptr, z4.dArray.Ptr) * inv_batch_size

    let dropout_matrix = createEmptyMatrix 1 1
    let dropout_matrix2 = createEmptyMatrix 1 1
        
    let gradient batch = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT keepin_rate weights batch 0.0f z1
        reluActivationModule.Apply(z1, z1) |> ignore
        //dropoutModule.Apply(z1,dropout_matrix,z1) |> ignore
        let z2 = sgemm2 T nT keepin_rate2 weights2 z1 0.0f z2
        reluActivationModule.Apply(z2, z2) |> ignore
        //dropoutModule2.Apply(z2,dropout_matrix2,z2) |> ignore
        let z3 = sgemm2 nT nT 1.0f weights2 z2 0.0f z3
        reluActivationModule.Apply(z3, z3) |> ignore
        let z4 = sgemm2 nT nT 1.0f weights z3 0.0f z4

        let squared_cost_error1 = binaryErrorModule.Apply(batch, z4, squared_cost_error1)
        sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error1 z3 1.0f weights |> ignore

        let squared_cost_error2 = sgemm2 T nT 1.0f weights squared_cost_error1 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, z3, squared_cost_error2) |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error2 z2 1.0f weights2 |> ignore

        let squared_cost_error3 = sgemm2 T nT 1.0f weights2 squared_cost_error2 0.0f squared_cost_error3
        binarySparseErrorModule.Apply(squared_cost_error3, z2, squared_cost_error3) |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) z1 squared_cost_error3 1.0f weights2 |> ignore

        let squared_cost_error4 = sgemm2 nT nT 1.0f weights2 squared_cost_error3 0.0f squared_cost_error4
        binarySparseErrorModule.Apply(squared_cost_error4, z1, squared_cost_error4) |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) batch squared_cost_error4 1.0f weights |> ignore


    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data)

    for epoch=1 to num_epochs do
        //fillRandomUniformMatrix 1.0f dropout_matrix
        //fillRandomUniformMatrix 1.0f dropout_matrix2
        for batch,_ in training_batches do
            gradient batch

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data)

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

let weights = createRandomUniformMatrix 784 784 1e-4f
let weights2 = createRandomUniformMatrix 784 512 1e-4f

let keepin1 = 0.35f
let keepin2 = 0.35f

relu_dropout_autoencoder weights 0.025f 5 training_batches dtest_data keepin1
let training_batches_sparse1, dtest_data_sparse1 = feedforward_sparse_pass weights training_batches dtest_data keepin1
relu_dropout_autoencoder weights2 0.025f 5 training_batches_sparse1 dtest_data_sparse1 keepin2

relu_dropout_autoencoder_fine_tune weights weights2 0.002f 100 training_batches dtest_data keepin1 keepin2

let training_batches_sparse2, dtest_data_sparse2 = feedforward_sparse_pass weights2 training_batches_sparse1 dtest_data_sparse1 keepin2

test_time training_batches_sparse2 dtest_data_sparse2
save_bitmap weights2
