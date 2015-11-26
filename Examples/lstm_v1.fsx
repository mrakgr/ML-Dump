// My first attempt at making an LSTM.

// I'll adapt the previous simple example though that won't be enough to test it.
// This will be just so I can familiarize myself with the process.

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

/// Logistic(x)
//let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>
let logisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> 
        if x <= -1.0f then 0.0f
        else if x >= 1.0f then 1.0f
        else 0.5f*x+0.5f @>

let cellUpdateModule = 
    new DeviceQuadraryTransformModule<float32> 
        <@ fun a b c d->  
        a*b+c*d @>

let elementwiseMultiplicationAndAdditionModule = 
    new DeviceTrinaryTransformModule<float32> 
        <@ fun a b c ->  
        a*b+c @>

let reluBlockOutputModule = 
    new DeviceBinaryTransformModule<float32> 
        <@ fun a b ->  
        let a_mod = if a > 0.0f then a else 0.0f
        a_mod*b @>

let errorOutputModule = 
    new DeviceTrinaryTransformModule<float32>
        <@ fun a b c ->
        let b_mod = if b > 0.0f then b else 0.0f 
        let c_mod = if c <= 0.0f || c >= 1.0f then 0.0f else 0.5f 
        a * b_mod * c_mod @>

let lstm_forward_block_input weights_input input weights_hidden prev_hidden hidden_bias function_output =
    sgemm2 T nT 1.0f weights_input input 0.0f function_output |> ignore
    sgemm2 T nT 1.0f weights_hidden prev_hidden 0.0f function_output |> ignore
    addBias function_output hidden_bias |> ignore
    reluActivationModule.Apply(function_output, function_output) |> ignore

let lstm_forward weights_input input weights_hidden prev_hidden weights_peephole cell_state hidden_bias function_output =
    sgemm2 T nT 1.0f weights_input input 0.0f function_output |> ignore
    sgemm2 T nT 1.0f weights_hidden prev_hidden 0.0f function_output |> ignore
    elementwiseMultiplicationAndAdditionModule.Apply(weights_peephole,cell_state,function_output,function_output) |> ignore
    addBias function_output hidden_bias |> ignore
    logisticActivationModule.Apply(function_output, function_output) |> ignore

let d_training_sequence1 = {num_rows=1; num_cols=1; dArray=worker.Malloc([|-1.0f|])}:dM
let d_training_sequence2 = {num_rows=1; num_cols=1; dArray=worker.Malloc([|-1.0f|])}:dM
let d_target_sequence = {num_rows=4; num_cols=1; dArray=worker.Malloc([|0.0f;0.0f;0.0f;1.0f|])}:dM

let hidden_size = 10

let weights_input_block = createRandomUniformMatrix 1 hidden_size 0.1f -0.05f
let weights_input_input = createRandomUniformMatrix 1 hidden_size 0.1f -0.05f
let weights_input_forget = createRandomUniformMatrix 1 hidden_size 0.1f -0.05f
let weights_input_output = createRandomUniformMatrix 1 hidden_size 0.1f -0.05f

let weights_hidden_block = createRandomUniformMatrix hidden_size hidden_size 0.1f -0.05f
let weights_hidden_input = createRandomUniformMatrix hidden_size hidden_size 0.1f -0.05f
let weights_hidden_forget = createRandomUniformMatrix hidden_size hidden_size 0.1f -0.05f
let weights_hidden_output = createRandomUniformMatrix hidden_size hidden_size 0.1f -0.05f

let bias_hidden_block = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
let bias_hidden_input = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
let bias_hidden_forget = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
let bias_hidden_output = createRandomUniformMatrix hidden_size 1 0.1f -0.05f

let weights_peephole_input = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
let weights_peephole_forget = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
let weights_peephole_output = createRandomUniformMatrix hidden_size 1 0.1f -0.05f

let weights_last = createRandomUniformMatrix hidden_size 4 0.1f -0.05f

let prev_hidden_state = createEmptyMatrix hidden_size d_training_sequence1.num_cols
setModule.Apply(0.0f, prev_hidden_state, prev_hidden_state) |> ignore

let cell_state = createEmptyMatrix hidden_size d_training_sequence1.num_cols
setModule.Apply(0.0f, prev_hidden_state, prev_hidden_state) |> ignore

let lstm_forward_block_input_allocate weights_input input weights_hidden prev_hidden hidden_bias =
    let function_output = sgemm T nT 1.0f weights_input input
    sgemm2 T nT 1.0f weights_hidden prev_hidden 0.0f function_output |> ignore
    addBias function_output hidden_bias |> ignore
    reluActivationModule.Apply(function_output, function_output)

let lstm_forward_allocate weights_input input weights_hidden prev_hidden weights_peephole cell_state hidden_bias =
    let function_output = sgemm T nT 1.0f weights_input input
    sgemm2 T nT 1.0f weights_hidden prev_hidden 0.0f function_output |> ignore
    elementwiseMultiplicationAndAdditionModule.Apply(weights_peephole,cell_state,function_output) |> ignore
    addBias function_output hidden_bias |> ignore
    logisticActivationModule.Apply(function_output, function_output)

let activation_block = lstm_forward_block_input_allocate weights_input_block d_training_sequence1 weights_hidden_block prev_hidden_state bias_hidden_block
let activation_input = lstm_forward_allocate weights_input_input d_training_sequence1 weights_hidden_input prev_hidden_state weights_peephole_input cell_state bias_hidden_input
let activation_forget = lstm_forward_allocate weights_input_forget d_training_sequence1 weights_hidden_forget prev_hidden_state weights_peephole_forget cell_state bias_hidden_forget
let cell_state_updated = cellUpdateModule.Apply(activation_block,activation_input,cell_state,activation_forget)
let activation_output = lstm_forward_allocate weights_input_output d_training_sequence1 weights_hidden_output prev_hidden_state weights_peephole_output cell_state_updated bias_hidden_output
let block_output = reluBlockOutputModule.Apply(cell_state_updated,activation_output)

// Just a little hack for easier debugging.    
let error_block_output = block_output
let error_output = errorOutputModule.Apply(error_block_output,cell_state_updated,activation_output)

let errorCellStateModule =
    new DeviceTrinaryTransformModule<float32>
        <@ fun a b c ->
        let c_mod = if c > 0.0f then 1.0f else 0.0f
        a*b*c_mod@>

let error_cell_state = errorCellStateModule.Apply(error_block_output,activation_output,cell_state_updated)
elementwiseMultiplicationAndAdditionModule.Apply(weights_peephole_output,error_output, error_cell_state, error_cell_state) |> ignore
// There should be 3 more such calls for error_cell_state, but I can skip them as I do not have errors from the upper layers.

let errorForgetModule =
    new DeviceTrinaryTransformModule<float32>
        <@ fun a b c ->
        let c_mod = if c <= 0.0f || c >= 1.0f then 0.0f else 0.5f 
        a*b*c_mod @>

let error_forget = errorForgetModule.Apply(error_cell_state,cell_state,activation_forget)

let errorInputModule = errorForgetModule
let error_input = errorInputModule.Apply(error_cell_state,cell_state,activation_block,activation_input)

// It just so happens that input block and the output have the same relu activation.
let errorBlockModule = errorCellStateModule

let error_block = errorBlockModule.Apply(error_cell_state,activation_input,activation_block)