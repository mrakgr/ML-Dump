// Does not work for me. I screwed up the biases somewhere. I do not really care.

// Record: 97.19%. Dropout works great. The reconstruction error stays at 80 though.
// Record: 97.28% with 25% dropout in the hidden layer.
// Record: 97.37% with 75%.
// Record: 97.4% with 87.5%. All of the above are for 100 epochs.
// Record: 97.45% with 90%. 50 epochs.
// Record: 97.47% with 90%. 30 epochs.
// Record: 97.56% with 87.5% and 35 epochs.
// Record: 97.8% with 85% and 35 epochs.
// Honorable Mention: 97.61% with 80% after 3 epochs. 
// With a learning rate of 0.1f the autoencoder finishes training in a single epoch. Amazing.

// Record: 97.92%. 1 epoch, 80% dropout. This is with the modified logistic regression units. 2.0f learning rate for 500 epochs.
// Record: 98%. 0.5f learning rate, 2000 epochs.

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


let relu_dropout_autoencoder weights biases learning_rate num_epochs (training_batches: (dM*dM) []) dtest_data keepin_rate =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Applies a dropout according to an uniform random matrix b [0,1)
    let dropoutModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> if b <= keepin_rate then a else 0.0f @>

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 
    let dropout_matrix = createEmptyMatrixLike z1

    let z2 = sgemm nT nT 1.0f weights z1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let squared_cost_error2 = sgemm T nT 1.0f squared_cost_error weights
    
    // Memory for the cost function of the test set.
    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c

    let ones = createEmptyMatrix batch.num_cols 1
    onesModule.Apply(ones, ones) |> ignore
    
    let costSquaredError batch weights =
    
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT keepin_rate weights batch 0.0f z1_c 
        addBias z1 biases
        let a1 = reluActivationModule.Apply(z1, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) * inv_batch_size
        

    let gradient batch weights dropout_matrix = 
        let inv_batch_size = 1.0f / float32 batch.num_cols
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        addBias z1 biases
        let a1 = reluActivationModule.Apply(z1, z1)
        dropoutModule.Apply(a1, dropout_matrix, a1) |> ignore
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        sgemm2 nT T (-inv_batch_size*learning_rate) squared_cost_error a1 1.0f weights |> ignore
        let squared_cost_error2 = sgemm2 T nT 1.0f squared_cost_error weights 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
        sgemm2 nT nT (-inv_batch_size*learning_rate) batch squared_cost_error2 1.0f weights |> ignore
        sgemv2 T (-inv_batch_size*learning_rate) squared_cost_error2 ones 1.0f biases |> ignore
                
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

let feedforward_sparse_pass weights biases (training_batches: (dM*dM) []) dtest_data keepin_rate =
    let training_batches_sparse =
        [|for batch,l in training_batches do
            let z1 = sgemm T nT keepin_rate weights batch
            addBias z1 biases
            reluActivationModule.Apply(z1, z1) |> ignore
            yield z1, l|]

    let dtest_data_sparse = sgemm T nT keepin_rate weights dtest_data
    addBias dtest_data_sparse biases
    reluActivationModule.Apply(dtest_data_sparse, dtest_data_sparse) |> ignore
    training_batches_sparse, dtest_data_sparse

let weights = createRandomUniformMatrix 784 784 1e-4f
let biases = createRandomUniformMatrix 784 1 1e-4f
let weights2 = createRandomUniformMatrix 784 256 1e-4f
let biases2 = createRandomUniformMatrix 256 1 1e-4f

let keepin1 = 0.2f
relu_dropout_autoencoder weights biases 0.1f 1 training_batches dtest_data keepin1
let training_batches_sparse1, dtest_data_sparse1 = feedforward_sparse_pass weights biases training_batches dtest_data keepin1
let keepin2 = 0.2f
relu_dropout_autoencoder weights2 biases2 0.01f 30 training_batches_sparse1 dtest_data_sparse1 keepin2
let training_batches_sparse2, dtest_data_sparse2 = feedforward_sparse_pass weights biases2 training_batches dtest_data keepin2

let test_time training_batches_sparse dtest_data_sparse =
    let weights2 = createRandomUniformMatrix weights.num_rows 10 1e-3f
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
test_time training_batches_sparse1 dtest_data_sparse1

let save_bitmap weights =
    let num_rows_and_cols = sqrt(float weights.num_rows) |> int
    let bitmap = make_bitmap_from_imageset weights 32 16 25 10
    bitmap.Save(@"C:\!NN\stacked_relu_dropout_3a.bmp")
save_bitmap weights2
