// It occured to me that I could just test it on Mnist.

// Let me try it. If the accuracy is below 90% for two layer, then I know something is really wrong.

// Record: 96.8%. Not to bad. I would have expected something like this for an LSTM in the feedforward case as 
// all those inter cell multiplication can't be making the optimization job easier...just what am I saying?

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
//[<ReflectedDefinition>]
//let inline steep_sigmoid x = 1.0f / (1.0f+exp(-3.0f*x))

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
//[<ReflectedDefinition>]
//let inline steep_sigmoid_derivative x = 3.0f * x * (1.0f - x)
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

    weights_peephole_input = createRandomMatrix hidden_size 1
    weights_peephole_forget = createRandomMatrix hidden_size 1
    weights_peephole_output = createRandomMatrix hidden_size 1

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
            broadcastingModule.BroadcastMultiply(weights_peephole,cell_state,function_output) |> ignore
            addBias function_output hidden_bias |> ignore
            logisticActivationModule.Apply(function_output, function_output)
        | Some input, None ->
            let function_output = dynamic_multiply T nT 1.0f weights_input input 0.0f function_output
            broadcastingModule.BroadcastMultiply(weights_peephole,cell_state,function_output) |> ignore
            addBias function_output hidden_bias |> ignore
            logisticActivationModule.Apply(function_output, function_output)
        | None, Some prev_hidden ->
            let function_output = dynamic_multiply T nT 1.0f weights_hidden prev_hidden 0.0f function_output
            broadcastingModule.BroadcastMultiply(weights_peephole,cell_state,function_output) |> ignore
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
    broadcastingModule.BroadcastMultiply(w.weights_peephole_output,error_output, error_cell_state) |> ignore

    match right with
        | Some (right_activations,right_pars, right_errors) -> 
            broadcastingModule.BroadcastMultiply(w.weights_peephole_input, right_errors.error_input, error_cell_state) |> ignore
            broadcastingModule.BroadcastMultiply(w.weights_peephole_forget, right_errors.error_forget, error_cell_state) |> ignore
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
    broadcastingModule.BroadcastMultiply(w.weights_peephole_output,error_output, error_cell_state) |> ignore

    match right with
        | Some (right_activations,right_pars, right_errors) -> 
            broadcastingModule.BroadcastMultiply(w.weights_peephole_input, right_errors.error_input, error_cell_state) |> ignore
            broadcastingModule.BroadcastMultiply(w.weights_peephole_forget, right_errors.error_forget, error_cell_state) |> ignore
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
            peepholeModule.ElementwiseMultiplyAndAverage(alpha,activations.cell_state_updated,errors_right.error_input,beta,grad.grad_weights_peephole_input) |> ignore
            peepholeModule.ElementwiseMultiplyAndAverage(alpha,activations.cell_state_updated,errors_right.error_forget,beta,grad.grad_weights_peephole_forget) |> ignore
        | None -> ()
    peepholeModule.ElementwiseMultiplyAndAverage(alpha,activations.cell_state_updated,errors.error_output,beta,grad.grad_weights_peephole_output) |> ignore

#load "load_mnist.fsx"
open Load_mnist.MnistLoad

let train = make_imageset trainSetData trainSetLabels
let test = make_imageset testSetData testSetLabels

/// The Mnist training set split into batches of 250.

/// I will also adjust the targets so that fall more into the linear part of tanh.
let batch_size = 125
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

        sgeam2 nT nT 0.5f dtrain_label 0.0f dtrain_label dtrain_label |> ignore

        yield (dtrain_data, dtrain_label)|]

let testing_batches =
    [|
    for i in 0..batch_size..test.num_images-1 do
        let s1 = test.num_rows*test.num_cols*i
        let s2 = test.num_rows*test.num_cols*(i+batch_size)-1
        let dtest_data: dM = 
                      {num_rows = test.num_rows * test.num_cols
                       num_cols = batch_size
                       dArray = worker.Malloc(test.float_data.[s1..s2])}

        if (dtest_data.num_cols*dtest_data.num_rows <> dtest_data.dArray.Length)
        then failwith "Invalid batch size (test)."

        let s1 = 10*i
        let s2 = 10*(i+batch_size)-1
        let dtest_label: dM =
                           {num_rows = 10
                            num_cols = batch_size
                            dArray = worker.Malloc(test.float_labels.[s1..s2])}
        if (dtest_label.num_cols*dtest_label.num_rows <> dtest_label.dArray.Length)
        then failwith "Invalid batch size (label)."

        sgeam2 nT nT 0.5f dtest_label 0.0f dtest_label dtest_label |> ignore

        yield (dtest_data, dtest_label)|]


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

let hidden_size = 1024
let batch,l = training_batches.[0]

let l1 = createRandomLstmCell hidden_size batch_size 784
let l2 = createRandomLstmCell l.num_rows batch_size hidden_size

let a1 = lstm_activation_allocation l1 (Some batch) None
let a2 = lstm_activation_allocation l2 (Some a1.block_output) None

let er_a2 = lstm_error_allocation_top_layer l a2.block_output None a2 l2
let er_a1 = lstm_error_allocation_middle_layer (Some (a2,l2,er_a2)) None a1 l1

let g1 = createGradsLike l1
let g2 = createGradsLike l2

let lstm_test num_iterations learning_coef momentum_rate =
    
    for i=1 to num_iterations do
        for batch, l in training_batches do

            let a1 = lstm_activation l1 (Some batch) None a1
            let a2 = lstm_activation l2 (Some a1.block_output) None a2

            let er_a2 = lstm_error_top_layer l a2.block_output None a2 l2 er_a2
            let er_a1 = lstm_error_middle_layer (Some (a2,l2,er_a2)) None a1 l1 er_a1

            weight_input_grads learning_coef er_a1 batch momentum_rate g1
            weight_input_grads learning_coef er_a2 a1.block_output 1.0f g2

            weight_biases_grad learning_coef er_a1 momentum_rate g1
            weight_biases_grad learning_coef er_a2 momentum_rate g2

            weight_peephole_grads learning_coef a1 None er_a1 momentum_rate g1
            weight_peephole_grads learning_coef a2 None er_a2 momentum_rate g2

            addGradsToWeights l1 g1
            addGradsToWeights l2 g2

        let costSquaredError batch l =
            let a1 = lstm_activation l1 (Some batch) None a1
            let a2 = lstm_activation l2 (Some a1.block_output) None a2

            squaredCostModule.Apply(l,a2.block_output) / float32 l.num_cols

        let calculate_validation_error() =
            let mutable c = 0.0f
            for batch,l in testing_batches do
                c <- c + (costSquaredError batch l)
            c / float32 testing_batches.Length

        let validation_error = calculate_validation_error()

        printfn "Validation error at epoch %i is %f" i validation_error


let learning_rate = 0.05f
let learning_coef = -learning_rate / float32 l.num_cols
let momentum_rate = 0.9f

lstm_test 50 learning_coef momentum_rate

// Finds the index of the max element of each column.
let rowReducer = new maxRowReduceModule<float32>()

let test_time() =
    let accuracyCalculate batch (l: dM) =
        let a1 = lstm_activation l1 (Some batch) None a1
        let a2 = lstm_activation l2 (Some a1.block_output) None a2

        let max_pred = rowReducer.Apply(a2.block_output)
        let max_labels = rowReducer.Apply(l)

        let pr,l = max_pred.Gather(), max_labels.Gather()

        let mutable c = 0
        for i=0 to pr.Length-1 do
            if pr.[i] = l.[i] then c <- c + 1
        c

    let calculate_validation_error() =
        let mutable c = 0
        for batch,l in testing_batches do
            c <- c + (accuracyCalculate batch l)
        c

    let accuracy = calculate_validation_error()
    printfn "The accuracy is %i/%i" accuracy 10000

test_time()