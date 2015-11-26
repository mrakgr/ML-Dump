// Does nowhere its full potential currently. I can only get 93-94% with it unlike 99%
// that was in the paper.

// Current record: 95.12%.

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

let mutable dtest_data: dM = 
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
// For computing the error in the final layer with the sparse activation function.
let trinarySparseErrorModule = new DeviceTrinaryTransformModule<float32> <@ fun y a c -> if c <> 0.0f then a-y else 0.0f @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

let hidden_layer_width = 1024

let sparse_autoencoder weights learning_rate num_epochs start_k min_k =
    let batch,_ = training_batches.[0]
    
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Preallocated memory. The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 

    // The sparse activation function module inspired by the k-sparse autoencoder.
    // http://arxiv.org/abs/1312.5663
    let sparseActivationModule = new sparsePiecewiseLinearActivationModule(z1.num_rows)

    let a1 = sparseActivationModule.Apply(z1,100)
    let z2 = sgemm nT nT 1.0f weights a1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let grad_second_layer = sgemm nT T inv_batch_size squared_cost_error z1
    let squared_cost_error2 = sgemm T nT 1.0f squared_cost_error weights
    binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
    let grad_first_layer = sgemm nT nT inv_batch_size batch squared_cost_error2 
    
    // Memory for the cost function of the test set.
    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c

    let costSquaredError batch weights k =
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c 
        let a1 = sparseActivationModule.Apply(z1,k, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) / float32 batch.num_cols

    let gradient batch weights k = 
        let inv_batch_size = 1.0f / float32 batch.num_cols

        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = sparseActivationModule.Apply(z1,k,a1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        let grad_second_layer = sgemm2 nT T inv_batch_size squared_cost_error a1 0.0f grad_second_layer
        let squared_cost_error2 = sgemm2 T nT 1.0f squared_cost_error weights 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
        let grad_first_layer = sgemm2 nT nT inv_batch_size batch squared_cost_error2 0.0f grad_first_layer

        /// Add the weight gradients together in the first layer
        sgeam2 nT nT 1.0f grad_second_layer 1.0f grad_first_layer grad_first_layer
                
    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data weights start_k)

    for epoch=1 to num_epochs do
        let current_k = max min_k (start_k-epoch+1)
        for batch,_ in training_batches do
            let grad = gradient batch weights current_k
            // Add them to the weights.
            sgeam2 nT nT 1.0f weights (-learning_rate) grad weights |> ignore

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data weights current_k)
        printfn "current_k is %i" current_k

/// Logistic(x)
let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>

/// sumall(map2(a*(log b) + (1.0f-a)*(1.0f - log b))
/// The logistic regression cost function.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule
                                <@ fun a b -> 
                                a*(log b) + (1.0f-a)*log (1.0f - b)@>

/// sumall(map(x*x)) for the L2 penalty
let squaredMapSum = new DeviceUnaryMapReduceModule <@ fun x -> x*x @>

let logistic_regression weights learning_rate lambda num_epochs = 
    let batch,_ = training_batches.[0]
    let inv_batch_size = 1.0f / float32 batch.num_cols

    let z1 = sgemm T nT 1.0f weights batch
    let weights_grad = sgemm nT T inv_batch_size batch z1

    let z1_c = sgemm T nT 1.0f weights dtest_data 

    let costLogRegression batch weights labels =
        let alpha = -1.0f/float32 batch.num_cols

        ///logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c
        let a1 = logisticActivationModule.Apply(z1,z1)

        /// alpha * sumall(labels*(log output) + (1.0f-labels)*log (1.0f - output))
        let cross_entropy_cost = alpha * crossEntropyCostModule.Apply(labels, a1)
        
        /// alpha * p.lambda * sumall(x*x)
        let reg_cost = alpha*lambda*squaredMapSum.Apply(weights)
        cross_entropy_cost+reg_cost

    let gradient batch weights labels =
        /// logistic(dWeights.T*dtrain_data)
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = logisticActivationModule.Apply(z1,z1)
        /// -(labels-a1)
        let cross_entropy_error = binaryErrorModule.Apply(labels,a1,a1)
        /// data * cross_entropy_error.T
        let weights_grad = sgemm2 nT T inv_batch_size batch cross_entropy_error 0.0f weights_grad
        // Constant for the L2 penalty.
        let weights_reg_const = 2.0f*lambda*inv_batch_size
        // Add the L2 penalty to the gradient.
        sgeam2 nT nT 1.0f weights_grad weights_reg_const weights weights_grad

    printfn "Cross entropy error of the logistic regression layer before optimization is %f" (costLogRegression dtest_data weights dtest_label)

    for epoch=1 to num_epochs do
        for batch,l in training_batches do
            let grad = gradient batch weights l
            // Add them to the weights.
            sgeam2 nT nT 1.0f weights (-learning_rate) grad weights |> ignore

        printfn "Cross entropy error of the logistic regression layer after epoch %i is %f" epoch (costLogRegression dtest_data weights dtest_label)
    

//let weights1 = createRandomUniformMatrix 784 10 1e-5f
let weights1 = createRandomUniformMatrix 784 hidden_layer_width 1e-5f
let weights2 = createRandomUniformMatrix (hidden_layer_width) 10 1e-5f
//let weights2 = createRandomUniformMatrix hidden_layer_width (hidden_layer_width-32) 1e-5f
//let weights3 = createRandomUniformMatrix (hidden_layer_width-32) 10 1e-5f
//let weights = load_weights_mnist @"C:\!NN\1000i-25k-sparse-weights.bin"

let k = 50

#time

sparse_autoencoder weights1 0.005f 600 100 k

let feedforward_sparse_pass weights =
    let batch, l = training_batches.[0]
    let z1 = sgemm T nT 1.0f weights batch
    let sparseActivationModule = new sparsePiecewiseLinearActivationModule(z1.num_rows)

    for i=0 to training_batches.Length-1 do
        let batch, l = training_batches.[i]
        let z1 = sgemm T nT 1.0f weights batch
        sparseActivationModule.Apply(z1,k,z1) |> ignore
        training_batches.[i] <- z1, l

    dtest_data <- sgemm T nT 1.0f weights dtest_data
    sparseActivationModule.Apply(dtest_data,k,dtest_data) |> ignore

feedforward_sparse_pass weights1

//sparse_autoencoder weights2 0.01f 200 100 k

//feedforward_sparse_pass weights2

logistic_regression weights2 0.005f 0.1f 4000
#time

let rowReducer = new maxRowReduceModule<float32>()

let predictions = sgemm T nT 1.0f weights2 dtest_data
let max_pred = rowReducer.Apply(predictions)
let max_labels = rowReducer.Apply(dtest_label)

let pr,l = max_pred.Gather(), max_labels.Gather()

let mutable c = 0
for i=0 to pr.Length-1 do
    if pr.[i] = l.[i] then c <- c + 1
printfn "The accuracy is %i/%i" c pr.Length

//save_weights @"C:\!NN\weights_784_1024_layer1.bin" weights1
//save_weights @"C:\!NN\weights_1024_992_layer1.bin" weights2
//save_weights @"C:\!NN\weights_992_10_layer1.bin" weights3

//let bitmap = make_bitmap_from_imageset weights1 28 28 40 25
//bitmap.Save(@"C:\!NN\sparse_weights_l1_1.bmp")