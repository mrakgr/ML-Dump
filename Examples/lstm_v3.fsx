// As I cannot make the other LSTM converge perfectly despite how simple the example is, I am going to try and
// modify the thing so the peepholes use the matrix product.

// https://github.com/cazala/synaptic
// If that does not work, I am going to try the library above.
// I've been getting nothing but installation problems with the Python libraries.
// In particular while I managed to instal Cgt, its code is very hard to read.

// The worst thing is that the lstm_v2 might be working properly.

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
//let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

// Relu activation.
//let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a > 0.0f then a else 0.0f @>

/// Logistic(x)
//let logisticActivationModule = new DeviceUnaryTransformModule<float32> <@ fun x -> 1.0f/(1.0f+exp(-x)) @>
(*
[<ReflectedDefinition>]
let inline clipped_linear_sigmoid x =
    if x <= -0.5f then 0.0f
    else if x >= 0.5f then 1.0f
    else x+0.5f
*)
[<ReflectedDefinition>]
let inline sigmoid x = 1.0f / (1.0f+exp(-x))

let logisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> sigmoid x @>

let cellUpdateModule = 
    new DeviceQuadraryTransformModule<float32> 
        <@ fun a b c d->  
        a*b+c*d @>

let elementwiseMultiplicationAndAdditionModule = 
    new DeviceTrinaryTransformModule<float32> 
        <@ fun a b c ->  
        a*b+c @>

(*
[<ReflectedDefinition>]
let inline clipped_linear_tanh a =
    if a < -1.0f then -1.0f 
    else if a > 1.0f then 1.0f
    else a
*)

//[<ReflectedDefinition>]
//let inline clipped_linear_relu a = if a > 0.0f then a else 0.0f

let tanhActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun a -> tanh a @>
    
/// Clipped linear tanh
let tanhBlockOutputModule = 
    new DeviceBinaryTransformModule<float32> 
        <@ fun a b ->  
        let a_mod = tanh a
        a_mod*b @>

//[<ReflectedDefinition>]
//let inline clipped_linear_sigmoid_derivative c = if c <= 0.0f || c >= 1.0f then 0.0f else 1.0f 
//[<ReflectedDefinition>]
//let inline clipped_linear_tanh_derivative c = if c <= -1.0f || c >= 1.0f then 0.0f else 1.0f 
//[<ReflectedDefinition>]
//let inline clipped_linear_relu_derivative c = if c > 0.0f then 1.0f else 0.0f 

[<ReflectedDefinition>]
let inline sigmoid_derivative c = c*(1.0f-c)
[<ReflectedDefinition>]
let inline tanh_derivative c = 1.0f-c*c

let errorOutputModule = 
    new DeviceTrinaryTransformModule<float32>
        <@ fun a b c ->
        let b_mod = tanh b
        let c_mod = sigmoid_derivative c
        a * b_mod * c_mod @>

let errorCellStateModule =
    new DeviceTrinaryTransformModule<float32>
        <@ fun a b c ->
        let c_mod = tanh_derivative c
        a*b*c_mod@>

let errorForgetModule =
    new DeviceTrinaryTransformModule<float32>
        <@ fun a b c ->
        let c_mod = sigmoid_derivative c
        a*b*c_mod @>

let d_training_sequence1 = Some ({num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;1.0f;1.0f|])}:dM)
let d_training_sequence2 = Some ({num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;1.0f;0.0f;1.0f|])}:dM)
let d_target_sequence = {num_rows=4; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;0.0f;0.5f;0.0f;0.0f;0.5f;0.0f;0.0f;0.5f;0.0f;0.0f;0.5f;0.0f;0.0f;0.0f|])}:dM

type lstmPars = {
    weights_input_block : dM
    weights_input_input : dM
    weights_input_forget : dM
    weights_input_output : dM

    weights_hidden_block : dM
    weights_hidden_input : dM
    weights_hidden_forget : dM
    weights_hidden_output : dM

    bias_hidden_block : dM
    bias_hidden_input : dM
    bias_hidden_forget : dM
    bias_hidden_output : dM

    weights_peephole_input : dM
    weights_peephole_forget : dM
    weights_peephole_output : dM

    cell_state : dM
    }

let createRandomLstmCell hidden_size hidden_state_width lower_layer_size =
    let createRandomMatrix a b =
        let scale = 1.0f/sqrt((a+b) |> float32)
        let location = -scale*0.5f
        createRandomUniformMatrix a b scale location
    let createEmptyAndSetMatrix a b set =
        let m = createEmptyMatrix a b
        setModule.Apply(set,m,m)
    {
    weights_input_block = createRandomMatrix lower_layer_size hidden_size
    weights_input_input = createRandomMatrix lower_layer_size hidden_size
    weights_input_forget = createRandomMatrix lower_layer_size hidden_size
    weights_input_output = createRandomMatrix lower_layer_size hidden_size

    weights_hidden_block = createRandomMatrix hidden_size hidden_size
    weights_hidden_input = createRandomMatrix hidden_size hidden_size
    weights_hidden_forget = createRandomMatrix hidden_size hidden_size
    weights_hidden_output = createRandomMatrix hidden_size hidden_size

    bias_hidden_block = createRandomMatrix hidden_size 1
    bias_hidden_input = createRandomMatrix hidden_size 1
    
    // The biases for the forget require special treatment for improved performance.
    // As recommended by Gers and Sutskever, initializing the gate to a high starting value should be good.
    bias_hidden_forget = createEmptyAndSetMatrix hidden_size 1 1.0f
    
    bias_hidden_output = createRandomMatrix hidden_size 1

    weights_peephole_input = createRandomMatrix hidden_size hidden_size
    weights_peephole_forget = createRandomMatrix hidden_size hidden_size
    weights_peephole_output = createRandomMatrix hidden_size hidden_size

    cell_state = createEmptyAndSetMatrix hidden_size hidden_state_width 0.0f
    }


let broadcastingModule = new broadcastingMultiplicationModule()

// Dynamically allocates memory if the matrix has not been used before.
let dynamic_multiply T1 T2 alpha weights input beta dest =
    match dest with
        | Some dest -> sgemm2 T1 T2 alpha weights input beta dest
        | None -> sgemm T1 T2 alpha weights input

let lstm_forward_block_input weights_input input weights_hidden prev_hidden hidden_bias function_output =
    match input, prev_hidden with
        | Some input, Some prev_hidden -> 
            let function_output = dynamic_multiply T nT 1.0f weights_input input 0.0f function_output
            sgemm2 T nT 1.0f weights_hidden prev_hidden 1.0f function_output |> ignore
            addBias function_output hidden_bias |> ignore
            tanhActivationModule.Apply(function_output, function_output)
        | Some input, None ->
            let function_output = dynamic_multiply T nT 1.0f weights_input input 0.0f function_output
            addBias function_output hidden_bias |> ignore
            tanhActivationModule.Apply(function_output, function_output)
        | None, Some prev_hidden ->
            let function_output = dynamic_multiply T nT 1.0f weights_hidden prev_hidden 0.0f function_output
            addBias function_output hidden_bias |> ignore
            tanhActivationModule.Apply(function_output, function_output)
        | None, None -> failwith "Invalid input to lstm_forward_block_input"

let lstm_forward weights_input input weights_hidden prev_hidden weights_peephole cell_state hidden_bias function_output =
    match input, prev_hidden with
        | Some input, Some prev_hidden -> 
            let function_output = dynamic_multiply T nT 1.0f weights_input input 0.0f function_output
            sgemm2 T nT 1.0f weights_hidden prev_hidden 1.0f function_output |> ignore
            sgemm2 T nT 1.0f weights_peephole cell_state 1.0f function_output |> ignore
            addBias function_output hidden_bias |> ignore
            logisticActivationModule.Apply(function_output, function_output)
        | Some input, None ->
            let function_output = dynamic_multiply T nT 1.0f weights_input input 0.0f function_output
            sgemm2 T nT 1.0f weights_peephole cell_state 1.0f function_output |> ignore
            addBias function_output hidden_bias |> ignore
            logisticActivationModule.Apply(function_output, function_output)
        | None, Some prev_hidden ->
            let function_output = dynamic_multiply T nT 1.0f weights_hidden prev_hidden 0.0f function_output
            sgemm2 T nT 1.0f weights_peephole cell_state 1.0f function_output |> ignore
            addBias function_output hidden_bias |> ignore
            logisticActivationModule.Apply(function_output, function_output)
        | None, None -> failwith "Invalid input to lstm_forward_block_input"


type lstmActivations = {
    activation_block : dM
    activation_input : dM
    activation_forget : dM
    cell_state_updated : dM
    activation_output : dM
    block_output : dM
    }

let lstm_activation_allocation (p : lstmPars) input prev_hidden_state =
    let activation_block = lstm_forward_block_input p.weights_input_block input p.weights_hidden_block prev_hidden_state p.bias_hidden_block None
    let activation_input = lstm_forward p.weights_input_input input p.weights_hidden_input prev_hidden_state p.weights_peephole_input p.cell_state p.bias_hidden_input None
    let activation_forget = lstm_forward p.weights_input_forget input p.weights_hidden_forget prev_hidden_state p.weights_peephole_forget p.cell_state p.bias_hidden_forget None
    let cell_state_updated = cellUpdateModule.Apply(activation_block,activation_input,p.cell_state,activation_forget)
    let activation_output = lstm_forward p.weights_input_output input p.weights_hidden_output prev_hidden_state p.weights_peephole_output cell_state_updated p.bias_hidden_output None
    let block_output = tanhBlockOutputModule.Apply(cell_state_updated,activation_output)
    { activation_block = activation_block; activation_input = activation_input; activation_forget = activation_forget; cell_state_updated = cell_state_updated; activation_output = activation_output; block_output = block_output}

let lstm_activation (p : lstmPars) input prev_hidden_state activations =
    let activation_block = lstm_forward_block_input p.weights_input_block input p.weights_hidden_block prev_hidden_state p.bias_hidden_block (Some activations.activation_block)
    let activation_input = lstm_forward p.weights_input_input input p.weights_hidden_input prev_hidden_state p.weights_peephole_input p.cell_state p.bias_hidden_input (Some activations.activation_input)
    let activation_forget = lstm_forward p.weights_input_forget input p.weights_hidden_forget prev_hidden_state p.weights_peephole_forget p.cell_state p.bias_hidden_forget (Some activations.activation_forget)
    let cell_state_updated = cellUpdateModule.Apply(activation_block,activation_input,p.cell_state,activation_forget,activations.cell_state_updated)
    let activation_output = lstm_forward p.weights_input_output input p.weights_hidden_output prev_hidden_state p.weights_peephole_output cell_state_updated p.bias_hidden_output (Some activations.activation_output)
    let block_output = tanhBlockOutputModule.Apply(cell_state_updated,activation_output,activations.block_output)
    { activation_block = activation_block; activation_input = activation_input; activation_forget = activation_forget; cell_state_updated = cell_state_updated; activation_output = activation_output; block_output = block_output}

type lstmErrors = {
    error_block_output : dM
    error_output : dM
    error_cell_state : dM
    error_forget : dM
    error_input : dM
    error_block : dM
    }

let lstm_error_allocation_cell error_block_output (right: (lstmActivations*lstmPars*lstmErrors) option) p w =
    let error_block_output =
        match error_block_output, right with
            | Some error_block_output, Some (_,right_pars, er_r) ->
                sgemm2 nT nT 1.0f w.weights_hidden_block er_r.error_block 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_input er_r.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_forget er_r.error_forget 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_output er_r.error_output 1.0f error_block_output
            | None, Some (_,right_pars, er_r) -> 
                let error_block_output = sgemm nT nT 1.0f w.weights_hidden_block er_r.error_block
                sgemm2 nT nT 1.0f w.weights_hidden_input er_r.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_forget er_r.error_forget 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_output er_r.error_output 1.0f error_block_output
            | Some error_block_output, None -> error_block_output
            | None, None -> failwith "Invalid input to lstm_error_allocation_cell"

    let error_output = errorOutputModule.Apply(error_block_output,p.cell_state_updated,p.activation_output)

    let error_cell_state = errorCellStateModule.Apply(error_block_output,p.activation_output,p.cell_state_updated)
    sgemm2 T nT 1.0f w.weights_peephole_output error_output 1.0f error_cell_state |> ignore

    match right with
        | Some (right_activations,right_pars, right_errors) -> 
            sgemm2 T nT 1.0f w.weights_peephole_input right_errors.error_input 1.0f error_cell_state |> ignore
            sgemm2 T nT 1.0f w.weights_peephole_forget right_errors.error_forget 1.0f error_cell_state |> ignore
            elementwiseMultiplicationAndAdditionModule.Apply(right_pars.cell_state, right_activations.activation_forget, error_cell_state, error_cell_state) |> ignore
        | None -> ()

    let error_forget = errorForgetModule.Apply(error_cell_state,w.cell_state,p.activation_forget)

    let errorInputModule = errorForgetModule
    let error_input = errorInputModule.Apply(error_cell_state,w.cell_state,p.activation_block)

    // It just so happens that input block and the output have the same tanh activation.
    let errorBlockModule = errorCellStateModule

    let error_block = errorBlockModule.Apply(error_cell_state,p.activation_input,p.activation_block)

    {
    error_block_output = error_block_output
    error_output = error_output
    error_cell_state = error_cell_state
    error_forget = error_forget
    error_input = error_input
    error_block = error_block
    }

let lstm_error_cell error_block_output (right: (lstmActivations*lstmPars*lstmErrors) option) p w e =
    let error_block_output =
        match error_block_output, right with
            | Some error_block_output, Some (_,right_pars, er_r) ->
                sgemm2 nT nT 1.0f w.weights_hidden_block er_r.error_block 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_input er_r.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_forget er_r.error_forget 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_output er_r.error_output 1.0f error_block_output
            | None, Some (_,right_pars, er_r) -> 
                let error_block_output = sgemm2 nT nT 1.0f w.weights_hidden_block er_r.error_block 0.0f e.error_block_output
                sgemm2 nT nT 1.0f w.weights_hidden_input er_r.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_forget er_r.error_forget 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f w.weights_hidden_output er_r.error_output 1.0f error_block_output
            | Some error_block_output, None -> error_block_output
            | None, None -> failwith "Invalid input to lstm_error_allocation_cell"

    let error_output = errorOutputModule.Apply(error_block_output,p.cell_state_updated,p.activation_output,e.error_output)

    let error_cell_state = errorCellStateModule.Apply(error_block_output,p.activation_output,p.cell_state_updated,e.error_cell_state)
    sgemm2 T nT 1.0f w.weights_peephole_output error_output 1.0f error_cell_state |> ignore

    match right with
        | Some (right_activations,right_pars, right_errors) -> 
            sgemm2 T nT 1.0f w.weights_peephole_input right_errors.error_input 1.0f error_cell_state |> ignore
            sgemm2 T nT 1.0f w.weights_peephole_forget right_errors.error_forget 1.0f error_cell_state |> ignore
            elementwiseMultiplicationAndAdditionModule.Apply(right_pars.cell_state, right_activations.activation_forget, error_cell_state, error_cell_state) |> ignore
        | None -> ()

    let error_forget = errorForgetModule.Apply(error_cell_state,w.cell_state,p.activation_forget,e.error_forget)

    let errorInputModule = errorForgetModule
    let error_input = errorInputModule.Apply(error_cell_state,w.cell_state,p.activation_block,e.error_input)

    // It just so happens that input block and the output have the same tanh activation.
    let errorBlockModule = errorCellStateModule

    let error_block = errorBlockModule.Apply(error_cell_state,p.activation_input,p.activation_block,e.error_block)

    {
    error_block_output = error_block_output
    error_output = error_output
    error_cell_state = error_cell_state
    error_forget = error_forget
    error_input = error_input
    error_block = error_block
    }

// For the top layer
let lstm_error_allocation_top_layer target output (right: (lstmActivations*lstmPars*lstmErrors) option) p w =
    let error_block_output = binaryErrorModule.Apply(target,output)
    lstm_error_allocation_cell (Some error_block_output) (right: (lstmActivations*lstmPars*lstmErrors) option) p w 

let lstm_error_top_layer target output (right: (lstmActivations*lstmPars*lstmErrors) option) p w e =
    let error_block_output = binaryErrorModule.Apply(target,output,e.error_block_output)
    lstm_error_cell (Some error_block_output) (right: (lstmActivations*lstmPars*lstmErrors) option) p w e

let lstm_error_allocation_middle_layer up (right: (lstmActivations*lstmPars*lstmErrors) option) p w =
    let error_block_output =
        match up with
            | Some (up_activations, up_pars, er_up) -> 
                let error_block_output = sgemm nT nT 1.0f up_pars.weights_input_block er_up.error_block
                sgemm2 nT nT 1.0f up_pars.weights_input_input er_up.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f up_pars.weights_input_forget er_up.error_forget 1.0f error_block_output |> ignore
                Some (sgemm2 nT nT 1.0f up_pars.weights_input_output er_up.error_output 1.0f error_block_output)
            | None -> None

    lstm_error_allocation_cell error_block_output (right: (lstmActivations*lstmPars*lstmErrors) option) p w 

let lstm_error_middle_layer up (right: (lstmActivations*lstmPars*lstmErrors) option) p w e =
    let error_block_output =
        match up with
            | Some (up_activations, up_pars, er_up) -> 
                let error_block_output = sgemm2 nT nT 1.0f up_pars.weights_input_block er_up.error_block 0.0f e.error_block_output
                sgemm2 nT nT 1.0f up_pars.weights_input_input er_up.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f up_pars.weights_input_forget er_up.error_forget 1.0f error_block_output |> ignore
                Some (sgemm2 nT nT 1.0f up_pars.weights_input_output er_up.error_output 1.0f error_block_output)
            | None -> None

    lstm_error_cell error_block_output (right: (lstmActivations*lstmPars*lstmErrors) option) p w e

type lstmGrads = {
    grad_weights_input_block : dM
    grad_weights_input_input : dM
    grad_weights_input_forget : dM
    grad_weights_input_output : dM

    grad_weights_hidden_block : dM
    grad_weights_hidden_input : dM
    grad_weights_hidden_forget : dM
    grad_weights_hidden_output : dM

    grad_bias_hidden_block : dM
    grad_bias_hidden_input : dM
    grad_bias_hidden_forget : dM
    grad_bias_hidden_output : dM

    grad_weights_peephole_input : dM
    grad_weights_peephole_forget : dM
    grad_weights_peephole_output : dM
    }

let createEmptyMatrixLikeAndSet m set =
    let t = createEmptyMatrixLike m
    setModule.Apply(set,t,t)

let createGradsLike (w: lstmPars) = {
    grad_weights_input_block = createEmptyMatrixLikeAndSet w.weights_input_block 0.0f
    grad_weights_input_input = createEmptyMatrixLikeAndSet w.weights_input_input 0.0f
    grad_weights_input_forget = createEmptyMatrixLikeAndSet w.weights_input_forget 0.0f
    grad_weights_input_output = createEmptyMatrixLikeAndSet w.weights_input_output 0.0f

    grad_weights_hidden_block = createEmptyMatrixLikeAndSet w.weights_hidden_block 0.0f
    grad_weights_hidden_input = createEmptyMatrixLikeAndSet w.weights_hidden_input 0.0f
    grad_weights_hidden_forget = createEmptyMatrixLikeAndSet w.weights_hidden_forget 0.0f
    grad_weights_hidden_output = createEmptyMatrixLikeAndSet w.weights_hidden_output 0.0f

    grad_bias_hidden_block = createEmptyMatrixLikeAndSet w.bias_hidden_block 0.0f
    grad_bias_hidden_input = createEmptyMatrixLikeAndSet w.bias_hidden_input 0.0f
    grad_bias_hidden_forget = createEmptyMatrixLikeAndSet w.bias_hidden_forget 0.0f
    grad_bias_hidden_output = createEmptyMatrixLikeAndSet w.bias_hidden_output 0.0f

    grad_weights_peephole_input = createEmptyMatrixLikeAndSet w.weights_peephole_input 0.0f
    grad_weights_peephole_forget = createEmptyMatrixLikeAndSet w.weights_peephole_forget 0.0f
    grad_weights_peephole_output = createEmptyMatrixLikeAndSet w.weights_peephole_output 0.0f
    }

let weight_input_grads alpha (errors: lstmErrors) input beta (g: lstmGrads) =
    sgemm2 nT T alpha input errors.error_block beta g.grad_weights_input_block |> ignore
    sgemm2 nT T alpha input errors.error_input beta g.grad_weights_input_input |> ignore
    sgemm2 nT T alpha input errors.error_forget beta g.grad_weights_input_forget |> ignore
    sgemm2 nT T alpha input errors.error_output beta g.grad_weights_input_output |> ignore

let weight_hidden_grads alpha errors_right block_output beta g =
    sgemm2 nT T alpha block_output errors_right.error_block beta g.grad_weights_hidden_block |> ignore
    sgemm2 nT T alpha block_output errors_right.error_input beta g.grad_weights_hidden_input |> ignore
    sgemm2 nT T alpha block_output errors_right.error_forget beta g.grad_weights_hidden_forget |> ignore
    sgemm2 nT T alpha block_output errors_right.error_output beta g.grad_weights_hidden_output |> ignore

// alpha would be the learning coeficient here usually
// beta would be the momentum_rate here usually
let weight_biases_grad alpha errors beta (g: lstmGrads) =
    calculateBias alpha errors.error_block beta g.grad_bias_hidden_block
    calculateBias alpha errors.error_input beta g.grad_bias_hidden_input
    calculateBias alpha errors.error_forget beta g.grad_bias_hidden_forget
    calculateBias alpha errors.error_output beta g.grad_bias_hidden_output

let peepholeModule = new elementwiseMultiplyAndAverageModule()

let weight_peephole_grads alpha activations errors_right errors beta grad =
    match errors_right with
        | Some errors_right ->
            sgemm2 nT T alpha activations.cell_state_updated errors_right.error_input beta grad.grad_weights_peephole_input |> ignore
            sgemm2 nT T alpha activations.cell_state_updated errors_right.error_forget beta grad.grad_weights_peephole_forget |> ignore
        | None -> ()
    sgemm2 nT T alpha activations.cell_state_updated errors.error_output beta grad.grad_weights_peephole_output |> ignore

let hidden_size = 50
let batch_size = d_training_sequence1.Value.num_cols

let l1 = createRandomLstmCell hidden_size batch_size 1
let l2 = createRandomLstmCell d_target_sequence.num_rows batch_size hidden_size

let a1 = lstm_activation_allocation l1 d_training_sequence1 None
let a2 = lstm_activation_allocation l1 d_training_sequence2 (Some a1.block_output)
let b2 = lstm_activation_allocation l2 (Some a2.block_output) None

let er_b2 = lstm_error_allocation_top_layer d_target_sequence b2.block_output None b2 l2
let er_a2 = lstm_error_allocation_middle_layer (Some (b2,l2,er_b2)) None a2 l1
let er_a1 = lstm_error_allocation_middle_layer None (Some (a2,l1,er_a2)) a1 l1

let g1 = createGradsLike l1
let g2 = createGradsLike l2

let addGradsToWeights lstm_weights lstm_grads =
    let w = lstm_weights
    let g = lstm_grads

    sgeam2 nT nT 1.0f w.weights_input_block 1.0f g.grad_weights_input_block w.weights_input_block |> ignore
    sgeam2 nT nT 1.0f w.weights_input_input 1.0f g.grad_weights_input_input w.weights_input_input |> ignore
    sgeam2 nT nT 1.0f w.weights_input_forget 1.0f g.grad_weights_input_forget w.weights_input_forget |> ignore
    sgeam2 nT nT 1.0f w.weights_input_output 1.0f g.grad_weights_input_output w.weights_input_output |> ignore

    sgeam2 nT nT 1.0f w.weights_hidden_block 1.0f g.grad_weights_hidden_block w.weights_hidden_block |> ignore
    sgeam2 nT nT 1.0f w.weights_hidden_input 1.0f g.grad_weights_hidden_input w.weights_hidden_input |> ignore
    sgeam2 nT nT 1.0f w.weights_hidden_forget 1.0f g.grad_weights_hidden_forget w.weights_hidden_forget |> ignore
    sgeam2 nT nT 1.0f w.weights_hidden_output 1.0f g.grad_weights_hidden_output w.weights_hidden_output |> ignore

    sgeam2 nT nT 1.0f w.weights_peephole_output 1.0f g.grad_weights_peephole_output w.weights_peephole_output |> ignore
    sgeam2 nT nT 1.0f w.weights_peephole_output 1.0f g.grad_weights_peephole_output w.weights_peephole_output |> ignore
    sgeam2 nT nT 1.0f w.weights_peephole_output 1.0f g.grad_weights_peephole_output w.weights_peephole_output |> ignore

    sgeam2 nT nT 1.0f w.bias_hidden_block 1.0f g.grad_bias_hidden_block w.bias_hidden_block |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_input 1.0f g.grad_bias_hidden_input w.bias_hidden_input |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_forget 1.0f g.grad_bias_hidden_forget w.bias_hidden_forget |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_output 1.0f g.grad_bias_hidden_output w.bias_hidden_output |> ignore

let lstm_test num_iterations learning_coef momentum_rate =
    
    for i=1 to num_iterations do

        let a1 = lstm_activation l1 d_training_sequence1 None a1
        let a2 = lstm_activation l1 d_training_sequence2 (Some a1.block_output) a2
        let b2 = lstm_activation l2 (Some a2.block_output) None b2

        let sq_er = squaredCostModule.Apply(d_target_sequence,b2.block_output) * 0.25f
        printfn "Squared error cost is %f at iteration %i" sq_er i

        let er_b2 = lstm_error_top_layer d_target_sequence b2.block_output None b2 l2 er_b2
        let er_a2 = lstm_error_middle_layer (Some (b2,l2,er_b2)) None a2 l1 er_a2
        let er_a1 = lstm_error_middle_layer None (Some (a2,l1,er_a2)) a1 l1 er_a1

        weight_input_grads learning_coef er_a1 d_training_sequence1.Value momentum_rate g1
        weight_input_grads learning_coef er_a2 d_training_sequence2.Value 1.0f g1
        weight_input_grads learning_coef er_b2 a2.block_output momentum_rate g2

        weight_hidden_grads learning_coef er_a2 a1.block_output momentum_rate g1

        weight_biases_grad learning_coef er_a1 momentum_rate g1
        weight_biases_grad learning_coef er_a2 1.0f g1
        weight_biases_grad learning_coef er_b2 momentum_rate g2

        weight_peephole_grads learning_coef a1 (Some er_a2) er_a1 momentum_rate g1
        weight_peephole_grads learning_coef a2 None er_a2 1.0f g1
        weight_peephole_grads learning_coef b2 None er_b2 momentum_rate g2

        addGradsToWeights l1 g1
        addGradsToWeights l2 g2

let inv_batch_size = d_training_sequence1.Value.num_cols |> float32
let learning_rate = 0.1f
let learning_coef = -inv_batch_size*learning_rate
let momentum_rate = 0.9f

lstm_test 500 learning_coef momentum_rate
