// First RNN with Rprop.

// Damn it. Like the momentum method, after a while it blows up.
// I am out of ideas for now. I've normalized the inputs, batched them, applied gradient clipping, momentum and lastly rprop.
// Either there is an error in my backprop process, or maybe I really do need an LSTM to memorize this simple dataset.
// The later seems just too improbable.

// It seems the problem was the lack of output nonlinear activations and improperly initialized weights.

// Does poorly on this example without biases.
// Very unstable with small hidden layer sizes.
// Converges much faster than SGD+Momentum.

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

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

// For errors in the middle layers using sparse activations.
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

// Relu activation.
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a > 0.0f then a else 0.0f @>

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
        weight_prev + sign_grad * delta @>

//let rng = System.Random()
//let training_sequence = [|([|-1.0f;-1.0f|],[|0.0f;0.0f;0.0f;1.0f|]);([|-1.0f;1.0f|],[|0.0f;0.0f;1.0f;0.0f|]);([|1.0f;-1.0f|],[|0.0f;1.0f;0.0f;0.0f|]);([|1.0f;1.0f|],[|1.0f;0.0f;0.0f;0.0f|])|]

//let d_training_sequence1 = [|for x in training_sequence -> {num_rows=1; num_cols=1; dArray=worker.Malloc([|(fst x).[0]|])}:dM |]
//let d_training_sequence2 = [|for x in training_sequence -> {num_rows=1; num_cols=1; dArray=worker.Malloc([|(fst x).[1]|])}:dM |]
//let d_target_sequence = [|for x in training_sequence -> {num_rows=4; num_cols=1; dArray=worker.Malloc(snd x)}:dM |]

let d_training_sequence1 = {num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;1.0f;1.0f|])}:dM
let d_training_sequence2 = {num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;1.0f;0.0f;1.0f|])}:dM
let d_target_sequence = {num_rows=4; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;0.0f|])}:dM

let batch1, batch2 = d_training_sequence1, d_training_sequence2
let target = d_target_sequence

let hidden_size = 10

let weights_input_hidden = createRandomUniformMatrix 1 hidden_size 0.1f -0.05f
let weights_hidden_hidden = createRandomUniformMatrix hidden_size hidden_size 0.1f -0.05f
let weights_hidden_output = createRandomUniformMatrix hidden_size 4 0.1f -0.05f

let bias_hidden = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
let bias_out = createRandomUniformMatrix 4 1 0.1f -0.05f

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

let delta_bias_out = createEmptyMatrixLike grad_bias_out
let delta_bias_hidden = createEmptyMatrixLike grad_bias_hidden
let delta_input_hidden = createEmptyMatrixLike grad_input_hidden
let delta_hidden_hidden = createEmptyMatrixLike grad_hidden_hidden
let delta_hidden_output = createEmptyMatrixLike grad_hidden_output

for x in [|delta_bias_out;delta_bias_hidden;delta_input_hidden;delta_hidden_hidden;delta_hidden_output|] do
    setModule.Apply(0.001f,x,x) |> ignore

let grad_bias_out_prev = createEmptyMatrixLike grad_bias_out
let grad_bias_hidden_prev = createEmptyMatrixLike grad_bias_hidden
let grad_input_hidden_prev = createEmptyMatrixLike grad_input_hidden
let grad_hidden_hidden_prev = createEmptyMatrixLike grad_hidden_hidden
let grad_hidden_output_prev = createEmptyMatrixLike grad_hidden_output

for x in [|grad_bias_out_prev;grad_bias_hidden_prev;grad_input_hidden_prev;grad_hidden_hidden_prev;grad_hidden_output_prev|] do
    setModule.Apply(0.0f,x,x) |> ignore

type gradPars = {
    grad_bias_out: dM
    grad_bias_hidden: dM
    grad_input_hidden: dM
    grad_hidden_hidden: dM
    grad_hidden_output: dM
    }

let use_bias = true

let rnn weights_input_hidden weights_hidden_hidden weights_hidden_output bias_hidden bias_out learning_rate momentum_rate num_epochs =
    
    let costSquaredError (batch1: dM) batch2 target =
        let inv_batch_size = 1.0f / float32 batch1.num_cols
        // Forward pass.

        let xh = sgemm2 T nT 1.0f weights_input_hidden batch1 0.0f xh
        let hh = sgemm2 T nT 1.0f weights_hidden_hidden prev_hidden_state  0.0f hh

        let combined_hidden = sgeam2 nT nT 1.0f xh 1.0f hh combined_hidden
        if use_bias then addBias combined_hidden bias_hidden
        reluActivationModule.Apply(combined_hidden,combined_hidden) |> ignore

        let xh2 = sgemm2 T nT 1.0f weights_input_hidden batch2 0.0f xh2
        let hh2 = sgemm2 T nT 1.0f weights_hidden_hidden combined_hidden 0.0f hh2

        let combined_hidden2 = sgeam2 nT nT 1.0f xh2 1.0f hh2 combined_hidden2
        if use_bias then addBias combined_hidden2 bias_hidden
        reluActivationModule.Apply(combined_hidden2,combined_hidden2) |> ignore

        let out = sgemm2 T nT 1.0f weights_hidden_output combined_hidden2 0.0f out
        if use_bias then addBias out bias_out

        reluActivationModule.Apply(out,out) |> ignore

        squaredCostModule.Apply(target, out) * inv_batch_size
        

    let gradient (batch1: dM) batch2 target (prev_grads: gradPars) (cur_grads: gradPars) = 
        let inv_batch_size = 1.0f / float32 batch1.num_cols

        let grad_bias_hidden = cur_grads.grad_bias_hidden
        let grad_bias_out = cur_grads.grad_bias_out
        let grad_input_hidden = cur_grads.grad_input_hidden
        let grad_hidden_hidden = cur_grads.grad_hidden_hidden
        let grad_hidden_output = cur_grads.grad_hidden_output

        let grad_bias_hidden_prev = prev_grads.grad_bias_hidden
        let grad_bias_out_prev = prev_grads.grad_bias_out
        let grad_input_hidden_prev = prev_grads.grad_input_hidden
        let grad_hidden_hidden_prev = prev_grads.grad_hidden_hidden
        let grad_hidden_output_prev = prev_grads.grad_hidden_output

        // Nesterov

        sgeam2 nT nT momentum_rate grad_bias_hidden 1.0f bias_hidden bias_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_bias_out 1.0f bias_out bias_out |> ignore

        sgeam2 nT nT momentum_rate grad_input_hidden 1.0f weights_input_hidden weights_input_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_hidden_hidden 1.0f weights_hidden_hidden weights_hidden_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_hidden_output 1.0f weights_hidden_output weights_hidden_output |> ignore

        // Forward pass.

        let xh = sgemm2 T nT 1.0f weights_input_hidden batch1 0.0f xh
        let hh = sgemm2 T nT 1.0f weights_hidden_hidden prev_hidden_state  0.0f hh

        let combined_hidden = sgeam2 nT nT 1.0f xh 1.0f hh combined_hidden
        if use_bias then addBias combined_hidden bias_hidden
        reluActivationModule.Apply(combined_hidden,combined_hidden) |> ignore

        let xh2 = sgemm2 T nT 1.0f weights_input_hidden batch2 0.0f xh2
        let hh2 = sgemm2 T nT 1.0f weights_hidden_hidden combined_hidden 0.0f hh2

        let combined_hidden2 = sgeam2 nT nT 1.0f xh2 1.0f hh2 combined_hidden2
        if use_bias then addBias combined_hidden2 bias_hidden
        reluActivationModule.Apply(combined_hidden2,combined_hidden2) |> ignore

        let out = sgemm2 T nT 1.0f weights_hidden_output combined_hidden2 0.0f out
        if use_bias then addBias out bias_out

        reluActivationModule.Apply(out,out) |> ignore

        // Backwards pass.

        let er_out = binaryErrorModule.Apply(target,out)
        
        let er_hidden2 = sgemm2 nT nT 1.0f weights_hidden_output er_out 0.0f er_hidden2
        binarySparseErrorModule.Apply(er_hidden2,er_hidden2) |> ignore

        let er_hidden = sgemm2 nT nT 1.0f weights_hidden_hidden er_hidden2 0.0f er_hidden
        binarySparseErrorModule.Apply(er_hidden,er_hidden) |> ignore

        // Remove Nesterov's momentum

        sgeam2 nT nT -momentum_rate grad_bias_hidden 1.0f bias_hidden bias_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_bias_out 1.0f bias_out bias_out |> ignore

        sgeam2 nT nT -momentum_rate grad_input_hidden 1.0f weights_input_hidden weights_input_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_hidden_hidden 1.0f weights_hidden_hidden weights_hidden_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_hidden_output 1.0f weights_hidden_output weights_hidden_output |> ignore

        // Calculate gradients

        sgemm2 nT T (-inv_batch_size*learning_rate) combined_hidden2 er_out momentum_rate grad_hidden_output |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) combined_hidden er_hidden2 momentum_rate grad_hidden_hidden |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) batch2 er_hidden2 momentum_rate grad_input_hidden |> ignore

        sgemm2 nT T (-inv_batch_size*learning_rate) prev_hidden_state er_hidden 1.0f grad_hidden_hidden |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) batch1 er_hidden 1.0f grad_input_hidden |> ignore

        calculateBias (-inv_batch_size*learning_rate) er_out momentum_rate grad_bias_out
        calculateBias (-inv_batch_size*learning_rate) er_hidden2 momentum_rate grad_bias_hidden
        calculateBias (-inv_batch_size*learning_rate) er_hidden 1.0f grad_bias_hidden

        // Adjust deltas

        rpropMinusModule.Apply(delta_bias_hidden,grad_bias_hidden_prev,grad_bias_hidden,delta_bias_hidden) |> ignore
        rpropMinusModule.Apply(delta_bias_out,grad_bias_out_prev,grad_bias_out,delta_bias_out) |> ignore
        rpropMinusModule.Apply(delta_input_hidden,grad_input_hidden_prev,grad_input_hidden,delta_input_hidden) |> ignore
        rpropMinusModule.Apply(delta_hidden_hidden,grad_hidden_hidden_prev,grad_hidden_hidden,delta_hidden_hidden) |> ignore
        rpropMinusModule.Apply(delta_hidden_output,grad_hidden_output_prev,grad_hidden_output,delta_hidden_output) |> ignore

        // Adjust weights

        rpropMinusWeightsAddModule.Apply(bias_hidden,grad_bias_hidden,delta_bias_hidden,bias_hidden) |> ignore
        rpropMinusWeightsAddModule.Apply(bias_out,grad_bias_out,delta_bias_out,bias_out) |> ignore
        rpropMinusWeightsAddModule.Apply(weights_input_hidden,grad_input_hidden,delta_input_hidden,weights_input_hidden) |> ignore
        rpropMinusWeightsAddModule.Apply(weights_hidden_hidden,grad_hidden_hidden,delta_hidden_hidden,weights_hidden_hidden) |> ignore
        rpropMinusWeightsAddModule.Apply(weights_hidden_output,grad_hidden_output,delta_hidden_output,weights_hidden_output) |> ignore

    //printfn "Square error cost of the reconstruction (before optimization) is %f" (costSquaredError batch1 batch2 target)

    let cur = {
        grad_bias_out = grad_bias_out
        grad_bias_hidden = grad_bias_hidden
        grad_input_hidden = grad_input_hidden
        grad_hidden_hidden = grad_hidden_hidden
        grad_hidden_output = grad_hidden_output
        } 

    let prev = {
        grad_bias_out = grad_bias_out_prev
        grad_bias_hidden = grad_bias_hidden_prev
        grad_input_hidden = grad_input_hidden_prev
        grad_hidden_hidden = grad_hidden_hidden_prev
        grad_hidden_output = grad_hidden_output_prev
        } 

    for iter=1 to num_epochs do
            let batch1 = d_training_sequence1
            let batch2 = d_training_sequence2
            let target = d_target_sequence

            gradient batch1 batch2 target prev cur
            gradient batch1 batch2 target cur prev

            printfn "Square error cost of the reconstruction after epoch %i is %f" iter (costSquaredError batch1 batch2 target)

// The learning rate does nothing for Rprop, but setting it to zero will zero out the weight matrices.
rnn weights_input_hidden weights_hidden_hidden weights_hidden_output bias_hidden bias_out 1.0f 0.f 50

