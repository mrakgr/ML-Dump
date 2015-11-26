(* 
I had a good idea instead of trying a k-sparse autoencoder instead to select 
activations by their standard deviation from the mean. It regularizes really
poorly though. I'll try the actual k-sparse autoencoder next.

Update:

let grad_second_layer = sgemm2 nT T inv_batch_size squared_cost_error z1 0.0f grad_second_layer

Turns out I fucked up the line above. z1 should be a1 in the above example.

*)

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

// The sparse activation module inspired by the k-sparse autoencoder.
// On Maxwell cards having the small block size of 32 is very efficient.
// http://arxiv.org/abs/1312.5663
type sparsePiecewiseLinearActivationModule(target) =
    inherit GPUModule(target)

    let grid_size = 384
    let block_size = 32

    new() = new sparsePiecewiseLinearActivationModule(GPUModuleTarget.Worker(worker))

    [<Kernel;ReflectedDefinition>]
    member this.Kernel (num_rows:int) (num_cols:int) (x:deviceptr<float32>) (y:deviceptr<float32>) (*(thresholds:deviceptr<float32>)*) (threshold_multiplier: float32) =
        let inline butterflyWarpReduce (value:float32) = 
            let v1 = value + __shfl_xor value 16 32
            let v2 = v1 + __shfl_xor v1 8 32
            let v3 = v2 + __shfl_xor v2 4 32
            let v4 = v3 + __shfl_xor v3 2 32
            v4 + __shfl_xor v4 1 32
        // Point block_start to where the column starts in the array.
        let mutable col = blockIdx.x

        while col < num_cols do
            // i is the row index
            let mutable row = threadIdx.x
            let mutable acc = 0.0f
            while row < num_rows do
                // idx is the absolute index in the array
                let idx = row + col * num_rows
                acc <- acc + x.[idx]
                // Increment the row index
                row <- row + blockDim.x

            __syncthreads()
            let column_mean = (butterflyWarpReduce acc) / (float32 num_rows)

            row <- threadIdx.x
            acc <- 0.0f
            while row < num_rows do
                // idx is the absolute index in the array
                let idx = row + col * num_rows
                // Accumulate the variances.
                acc <- acc + (x.[idx]-column_mean)*(x.[idx]-column_mean)
                // Increment the row index
                row <- row + blockDim.x

            __syncthreads()
            let variance_sum = (butterflyWarpReduce acc) / (float32 num_rows)

            let standard_deviation = sqrt(variance_sum)
            let threshold = column_mean + standard_deviation*threshold_multiplier

            row <- threadIdx.x
            while row < num_rows do
                // idx is the absolute index in the array
                let idx = row + col * num_rows
                
                // Let the function activate if it is above the threshold.
                y.[idx] <- if x.[idx] > threshold then x.[idx] else 0.0f

                // Increment the row index
                row <- row + blockDim.x

            //if threadIdx.x = 0 then thresholds.[col] <- threshold

            col <- col + gridDim.x

    member this.Apply((dmat: dM), (threshold_mutliplier:float32), (rmat: dM)(*, (thresholds: dM)*)) =
        if dmat.dArray.Length <> rmat.dArray.Length then failwith "dmat.dArray.Length <> rmat.dArray.Length in sparsePiecewiseLinearActivationModule"
        let lp = LaunchParam(min grid_size dmat.num_rows, block_size)
        this.GPULaunch <@ this.Kernel @> lp dmat.num_rows dmat.num_cols dmat.dArray.Ptr rmat.dArray.Ptr (*thresholds.dArray.Ptr*) threshold_mutliplier
        (rmat: dM)//, (thresholds: dM)

    member this.Apply((dmat: dM), (threshold_mutliplier:float32)) =
        let lp = LaunchParam(min grid_size dmat.num_rows, block_size)
        let rmat = createEmptyMatrixLike dmat
        //let thresholds = createEmptyMatrix dmat.num_cols 1
        this.GPULaunch <@ this.Kernel @> lp dmat.num_rows dmat.num_cols dmat.dArray.Ptr rmat.dArray.Ptr (*thresholds.dArray.Ptr*) threshold_mutliplier
        (rmat: dM)//, (thresholds: dM)


// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>
// For computing the error in the final layer with the sparse activation function.
let trinarySparseErrorModule = new DeviceTrinaryTransformModule<float32> <@ fun y a c -> if c <> 0.0f then a-y else 0.0f @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>
// The sparse activation function module inspired by the k-sparse autoencoder.
// http://arxiv.org/abs/1312.5663
let sparseActivationModule = new sparsePiecewiseLinearActivationModule()


let sparse_autoencoder learning_rate num_epochs =
    let weights = createRandomUniformMatrix 784 1000 1e-5f

    let batch,_ = training_batches.[0]
    let inv_batch_size = 1.0f / float32 batch.num_cols

    // Preallocated memory.The modules also get compiled the first time they are run.
    let z1 = sgemm T nT 1.0f weights batch 
    let a1 = sparseActivationModule.Apply(z1,0.5f)
    let z2 = sgemm nT nT 1.0f weights a1
    let squared_cost_error = binaryErrorModule.Apply(batch, z2)
    let grad_second_layer = sgemm nT T inv_batch_size squared_cost_error z1
    let squared_cost_error2 = sgemm T nT 1.0f squared_cost_error weights
    //binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
    let grad_first_layer = sgemm nT nT inv_batch_size batch squared_cost_error2 

    // Memory for the cost function of the test set.
    let z1_c = sgemm T nT 1.0f weights dtest_data 
    let z2_c = sgemm nT nT 1.0f weights z1_c

    let costSquaredError batch weights threshold_multiplier =
        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1_c 
        let a1 = sparseActivationModule.Apply(z1,threshold_multiplier, z1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2_c

        squaredCostModule.Apply(batch, z2) / float32 batch.num_cols

    let gradient batch weights threshold_multiplier = 
        let inv_batch_size = 1.0f / float32 batch.num_cols

        let z1 = sgemm2 T nT 1.0f weights batch 0.0f z1
        let a1 = sparseActivationModule.Apply(z1,threshold_multiplier,a1)
        let z2 = sgemm2 nT nT 1.0f weights a1 0.0f z2

        let squared_cost_error = binaryErrorModule.Apply(batch, z2, squared_cost_error)
        let grad_second_layer = sgemm2 nT T inv_batch_size squared_cost_error z1 0.0f grad_second_layer
        let squared_cost_error2 = sgemm2 T nT 1.0f squared_cost_error weights 0.0f squared_cost_error2
        binarySparseErrorModule.Apply(squared_cost_error2, a1, squared_cost_error2) |> ignore
        let grad_first_layer = sgemm2 nT nT inv_batch_size batch squared_cost_error2 0.0f grad_first_layer

        /// Add the weight gradients together in the first layer
        sgeam2 nT nT 1.0f grad_second_layer 1.0f grad_first_layer grad_first_layer
                
    // In standard deviations.
    let start_threshold_multiplier = 0.0f
    printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError dtest_data weights start_threshold_multiplier)

    for epoch=1 to num_epochs do
        let current_threshold_multiplier = min ((float32 epoch)*0.1f+start_threshold_multiplier) 1.0f
        for batch,_ in training_batches do
            let grad = gradient batch weights current_threshold_multiplier
            // Add them to the weights.
            sgeam2 nT nT 1.0f weights (-learning_rate) grad weights |> ignore

        printfn "Square error cost of the reconstruction after epoch %i is %f" epoch (costSquaredError dtest_data weights current_threshold_multiplier)
        printfn "current_threshold_multiplier is %f" current_threshold_multiplier
    weights
        
#time
let weights = sparse_autoencoder 0.01f 200
#time

let batch,_ = training_batches.[0]

let z1 = sgemm T nT 1.0f weights batch 
let a1 = sparseActivationModule.Apply(z1,1.0f, z1)
let z2 = sgemm nT nT 1.0f weights a1 

let bitmap = make_bitmap_from_imageset weights 28 28 40 25
bitmap.Save(@"C:\!NN\sparse_weights8.bmp")