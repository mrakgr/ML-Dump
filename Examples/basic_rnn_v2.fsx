// As I am using the basic_rnn to test the unfolded_rnn, the addition non-associativty is messing with me.
// I am going to try various thing to see if the result changes.
// I am not sure whether to interpret the difference between unfolded_rnn and basic_rnn as errors.


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

// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>
squaredCostModule.GPUForceLoad()

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

// Relu activation.
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a > 0.0f then a else 0.0f @>

// Gradient clipping module.
let gradclipModule = 
    new DeviceUnaryCoefTransformModule<float32>
        <@ fun coef_a a ->
        if a > coef_a then coef_a
        else if a < -coef_a then -coef_a
        else a@>

let rng = System.Random()
let training_sequence = [|([|0.0f;0.0f|],[|0.0f;0.0f;0.0f;1.0f|]);([|0.0f;1.0f|],[|0.0f;0.0f;1.0f;0.0f|]);([|1.0f;0.0f|],[|0.0f;1.0f;0.0f;0.0f|]);([|1.0f;1.0f|],[|1.0f;0.0f;0.0f;0.0f|])|]

//let d_training_sequence1 = [|for x in training_sequence -> {num_rows=1; num_cols=1; dArray=worker.Malloc([|(fst x).[0]|])}:dM |]
//let d_training_sequence2 = [|for x in training_sequence -> {num_rows=1; num_cols=1; dArray=worker.Malloc([|(fst x).[1]|])}:dM |]
//let d_target_sequence = [|for x in training_sequence -> {num_rows=4; num_cols=1; dArray=worker.Malloc(snd x)}:dM |]

let d_training_sequence1 = {num_rows=1; num_cols=4; dArray=worker.Malloc([|-1.0f;-1.0f;1.0f;1.0f|])}:dM
let d_training_sequence2 = {num_rows=1; num_cols=4; dArray=worker.Malloc([|-1.0f;1.0f;-1.0f;1.0f|])}:dM
let d_target_sequence = {num_rows=4; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;0.0f|])}:dM

let batch1, batch2 = d_training_sequence1, d_training_sequence2
let target = d_target_sequence

let hidden_size = 250
(*
let weights_input_hidden = createRandomUniformMatrix 1 hidden_size 0.1f -0.05f
let weights_hidden_hidden = createRandomUniformMatrix hidden_size hidden_size 0.1f -0.05f
let weights_hidden_output = createRandomUniformMatrix hidden_size 4 0.1f -0.05f

let bias_hidden = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
let bias_out = createRandomUniformMatrix 4 1 0.1f -0.05f

save_weights @"C:\Temp\weights_input_hidden" weights_input_hidden
save_weights @"C:\Temp\weights_hidden_hidden" weights_hidden_hidden
save_weights @"C:\Temp\weights_hidden_output" weights_hidden_output
save_weights @"C:\Temp\bias_hidden" bias_hidden
save_weights @"C:\Temp\bias_out" bias_out
*)
let weights_input_hidden = load_weights_mnist @"C:\Temp\weights_input_hidden" 1
let weights_hidden_hidden = load_weights_mnist @"C:\Temp\weights_hidden_hidden" hidden_size
let weights_hidden_output = load_weights_mnist @"C:\Temp\weights_hidden_output" hidden_size
let bias_hidden = load_weights_mnist @"C:\Temp\bias_hidden" hidden_size
let bias_out = load_weights_mnist @"C:\Temp\bias_out" 4

let prev_hidden_state = createEmptyMatrix hidden_size d_training_sequence1.num_cols
setModule.Apply(0.0f, prev_hidden_state, prev_hidden_state) |> ignore

// Forward pass.
let xh = sgemm T nT 1.0f weights_input_hidden batch1 
let hh = sgemm T nT 1.0f weights_hidden_hidden prev_hidden_state 

let combined_hidden = sgeam nT nT 1.0f xh 1.0f hh
addBias combined_hidden bias_hidden

let xh2 = sgemm T nT 1.0f weights_input_hidden batch2 
let hh2 = sgemm T nT 1.0f weights_hidden_hidden combined_hidden 

let combined_hidden2 = sgeam nT nT 1.0f xh2 1.0f hh2
addBias combined_hidden2 bias_hidden

let out = sgemm T nT 1.0f weights_hidden_output combined_hidden2
addBias out bias_out

// Backwards pass.

let er_out = binaryErrorModule.Apply(target,out)
let grad_hidden_output = sgemm nT T 1.0f combined_hidden er_out

let er_hidden2 = sgemm nT nT 1.0f weights_hidden_output er_out
let grad_hidden_hidden = sgemm nT T 1.0f combined_hidden er_hidden2
let grad_input_hidden = sgemm nT T 1.0f batch2 er_hidden2

let er_hidden = sgemm nT nT 1.0f weights_hidden_hidden er_hidden2

sgemm2 nT T 1.0f prev_hidden_state er_hidden 1.0f grad_hidden_hidden |> ignore
sgemm2 nT T 1.0f batch1 er_hidden 1.0f grad_input_hidden |> ignore

setModule.Apply(0.0f,grad_input_hidden,grad_input_hidden) |> ignore
setModule.Apply(0.0f,grad_hidden_hidden,grad_hidden_hidden) |> ignore
setModule.Apply(0.0f,grad_hidden_output,grad_hidden_output) |> ignore

let grad_bias_out = createEmptyMatrixLike bias_out
let grad_bias_hidden = createEmptyMatrixLike bias_hidden

setModule.Apply(0.0f,grad_bias_out,grad_bias_out) |> ignore
setModule.Apply(0.0f,grad_bias_hidden,grad_bias_hidden) |> ignore

let use_bias = true

let rnn weights_input_hidden weights_hidden_hidden weights_hidden_output learning_rate momentum_rate num_epochs =
    
    let costSquaredError (batch1: dM) batch2 target =
        let inv_batch_size = 1.0f / float32 batch1.num_cols
        // Forward pass.

        let combined_hidden = sgemm2 T nT 1.0f weights_input_hidden batch1 0.0f combined_hidden
        if use_bias then addBias combined_hidden bias_hidden
        reluActivationModule.Apply(combined_hidden,combined_hidden) |> ignore

        let combined_hidden2 = sgemm2 T nT 1.0f weights_hidden_hidden combined_hidden 0.0f combined_hidden2
        let xh2 = sgemm2 T nT 1.0f weights_input_hidden batch2 1.0f combined_hidden2
        if use_bias then addBias combined_hidden2 bias_hidden
        reluActivationModule.Apply(combined_hidden2,combined_hidden2) |> ignore

        let out = sgemm2 T nT 1.0f weights_hidden_output combined_hidden2 0.0f out
        if use_bias then addBias out bias_out

        reluActivationModule.Apply(out,out) |> ignore

        squaredCostModule.Apply(target, out) * inv_batch_size
        

    let gradient (batch1: dM) batch2 target = 
        let inv_batch_size = 1.0f / float32 batch1.num_cols

        // Nesterov
        (*
        sgeam2 nT nT momentum_rate grad_bias_hidden 1.0f bias_hidden bias_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_bias_out 1.0f bias_out bias_out |> ignore

        sgeam2 nT nT momentum_rate grad_input_hidden 1.0f weights_input_hidden weights_input_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_hidden_hidden 1.0f weights_hidden_hidden weights_hidden_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_hidden_output 1.0f weights_hidden_output weights_hidden_output |> ignore
        *)

        // Forward pass.

        let combined_hidden = sgemm2 T nT 1.0f weights_input_hidden batch1 0.0f combined_hidden
        if use_bias then addBias combined_hidden bias_hidden
        reluActivationModule.Apply(combined_hidden,combined_hidden) |> ignore

        let combined_hidden2 = sgemm2 T nT 1.0f weights_hidden_hidden combined_hidden 0.0f combined_hidden2
        let xh2 = sgemm2 T nT 1.0f weights_input_hidden batch2 1.0f combined_hidden2
        if use_bias then addBias combined_hidden2 bias_hidden
        reluActivationModule.Apply(combined_hidden2,combined_hidden2) |> ignore

        let out = sgemm2 T nT 1.0f weights_hidden_output combined_hidden2 0.0f out
        if use_bias then addBias out bias_out

        reluActivationModule.Apply(out,out) |> ignore

        let sq_er = squaredCostModule.Apply(d_target_sequence,out) * inv_batch_size
        printfn "Squared error cost is %f at iteration " sq_er

        // Backwards pass.

        let er_out = binaryErrorModule.Apply(target, out, er_out)
        
        let er_hidden2 = sgemm2 nT nT 1.0f weights_hidden_output er_out 0.0f er_hidden2
        binarySparseErrorModule.Apply(er_hidden2,er_hidden2) |> ignore

        let er_hidden = sgemm2 nT nT 1.0f weights_hidden_hidden er_hidden2 0.0f er_hidden
        binarySparseErrorModule.Apply(er_hidden,er_hidden) |> ignore

        // Remove Nesterov's momentum
        (*
        sgeam2 nT nT -momentum_rate grad_bias_hidden 1.0f bias_hidden bias_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_bias_out 1.0f bias_out bias_out |> ignore

        sgeam2 nT nT -momentum_rate grad_input_hidden 1.0f weights_input_hidden weights_input_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_hidden_hidden 1.0f weights_hidden_hidden weights_hidden_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_hidden_output 1.0f weights_hidden_output weights_hidden_output |> ignore
        *)
        // Calculate gradients

        let grad_hidden_output = sgemm2 nT T (-inv_batch_size*learning_rate) combined_hidden2 er_out momentum_rate grad_hidden_output
        let grad_hidden_hidden = sgemm2 nT T (-inv_batch_size*learning_rate) combined_hidden er_hidden2 momentum_rate grad_hidden_hidden
        let grad_input_hidden = sgemm2 nT T (-inv_batch_size*learning_rate) batch2 er_hidden2 momentum_rate grad_input_hidden

        //sgemm2 nT T (-inv_batch_size*learning_rate) prev_hidden_state er_hidden 1.0f grad_hidden_hidden |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) batch1 er_hidden 1.0f grad_input_hidden |> ignore

        calculateBias (-inv_batch_size*learning_rate) er_out momentum_rate grad_bias_out
        calculateBias (-inv_batch_size*learning_rate) er_hidden2 momentum_rate grad_bias_hidden
        calculateBias (-inv_batch_size*learning_rate) er_hidden 1.0f grad_bias_hidden

        // Gradient clipping.
        //for x in [|grad_hidden_output;grad_hidden_hidden;grad_input_hidden;grad_bias_hidden;grad_bias_out|] do
            //gradclipModule.Apply(1.0f,x,x) |> ignore

        // Adjust weights

        sgeam2 nT nT 1.0f weights_hidden_output 1.0f grad_hidden_output weights_hidden_output |> ignore
        sgeam2 nT nT 1.0f weights_hidden_hidden 1.0f grad_hidden_hidden weights_hidden_hidden |> ignore
        sgeam2 nT nT 1.0f weights_input_hidden 1.0f grad_input_hidden weights_input_hidden |> ignore

        sgeam2 nT nT 1.0f bias_hidden 1.0f grad_bias_hidden bias_hidden |> ignore
        sgeam2 nT nT 1.0f bias_out 1.0f grad_bias_out bias_out |> ignore


    //printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError batch1 batch2 target)

    for iter=1 to num_epochs do
            let batch1 = d_training_sequence1
            let batch2 = d_training_sequence2
            let target = d_target_sequence

            gradient batch1 batch2 target

            //printfn "Square error cost of the reconstruction after epoch %i is %f" iter (costSquaredError batch1 batch2 target)

rnn weights_input_hidden weights_hidden_hidden weights_hidden_output 0.1f 0.9f 31
