// I just realized that the reason why Reber grammar blows up is because of all the memory conficts.
// Both rnn_standard and rnn_lstm are complete failures from a design standpoint. I need to remake them.

// v2 is a remake without all the dynamic memory tomfollery.

#load "rnn_standard_v2.fsx"
open Rnn_standard_v2
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

let cellUpdateModule = 
    new DeviceQuadraryTransformModule<float32> 
        <@ fun a b c d->  
        a*b+c*d @>

let elementwiseMultiplicationModule = 
    new DeviceBinaryTransformModule<float32> 
        <@ fun a b ->  
        a*b @>

let elementwiseMultiplicationAndAdditionModule = 
    new DeviceTrinaryTransformModule<float32> 
        <@ fun a b c ->  
        a*b+c @>
   
let tanhBlockOutputModule = 
    new DeviceBinaryTransformModule<float32> 
        <@ fun a b ->  
        let a_mod = tanh a
        a_mod*b @>

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

    }

let createRandomLstmCell hidden_size lower_layer_size =
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
    }


let broadcastingModule = new broadcastingMultiplicationModule()

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

let lstm_forward weights_input input weights_hidden prev_hidden weights_peephole (cell_state: dM option) hidden_bias function_output =
    match input, prev_hidden with
        | Some input, Some prev_hidden -> 
            let function_output = dynamic_multiply T nT 1.0f weights_input input 0.0f function_output
            sgemm2 T nT 1.0f weights_hidden prev_hidden 1.0f function_output |> ignore
            broadcastingModule.BroadcastMultiply(weights_peephole,cell_state.Value,function_output) |> ignore
            addBias function_output hidden_bias |> ignore
            logisticActivationModule.Apply(function_output, function_output)
        | Some input, None ->
            let function_output = dynamic_multiply T nT 1.0f weights_input input 0.0f function_output
            addBias function_output hidden_bias |> ignore
            logisticActivationModule.Apply(function_output, function_output)
        | None, Some prev_hidden ->
            let function_output = dynamic_multiply T nT 1.0f weights_hidden prev_hidden 0.0f function_output
            broadcastingModule.BroadcastMultiply(weights_peephole,cell_state.Value,function_output) |> ignore
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

let lstm_activation (p : lstmPars) input prev_activations activations =
    let prev_hidden_state, prev_cell_state = 
        match prev_activations with
            | Some x -> Some x.block_output, Some x.cell_state_updated
            | None -> None, None

    let cur_activation_block, cur_activation_input, cur_activation_forget, cur_cell_state_updated, cur_activation_output, cur_block_output =
        match activations with
            | Some x -> (Some x.activation_block), (Some x.activation_input), (Some x.activation_forget), (Some x.cell_state_updated), (Some x.activation_output), (Some x.block_output)
            | None -> None, None, None, None, None, None

    let activation_block = lstm_forward_block_input p.weights_input_block input p.weights_hidden_block prev_hidden_state p.bias_hidden_block cur_activation_block
    let activation_input = lstm_forward p.weights_input_input input p.weights_hidden_input prev_hidden_state p.weights_peephole_input prev_cell_state p.bias_hidden_input cur_activation_input
    let activation_forget = lstm_forward p.weights_input_forget input p.weights_hidden_forget prev_hidden_state p.weights_peephole_forget prev_cell_state p.bias_hidden_forget cur_activation_forget
    
    let cell_state_updated =
        match prev_cell_state, cur_cell_state_updated with
            | Some prev_cell_state, Some cur_cell_state_updated ->
                cellUpdateModule.Apply(activation_block,activation_input,prev_cell_state,activation_forget,cur_cell_state_updated)
            | Some prev_cell_state, None ->
                cellUpdateModule.Apply(activation_block,activation_input,prev_cell_state,activation_forget)
            | None, Some cur_cell_state_updated ->
                elementwiseMultiplicationModule.Apply(activation_block,activation_input,cur_cell_state_updated)
            | None, None ->
                elementwiseMultiplicationModule.Apply(activation_block,activation_input)
    
    let activation_output = lstm_forward p.weights_input_output input p.weights_hidden_output prev_hidden_state p.weights_peephole_output (Some cell_state_updated) p.bias_hidden_output cur_activation_output

    let block_output = 
        match cur_block_output with
            | Some cur_block_output ->
                tanhBlockOutputModule.Apply(cell_state_updated,activation_output,cur_block_output)
            | None ->
                tanhBlockOutputModule.Apply(cell_state_updated,activation_output)
    { activation_block = activation_block; activation_input = activation_input; activation_forget = activation_forget; cell_state_updated = cell_state_updated; activation_output = activation_output; block_output = block_output}
    
type lstmErrors = {
    error_block_output : dM
    error_output : dM
    error_cell_state : dM
    error_forget : dM
    error_input : dM
    error_block : dM
    }

let createErrorsLikeActivationsLSTM act =
    { error_block_output = createEmptyMatrixLike act.block_output; error_output = createEmptyMatrixLike act.activation_output; error_cell_state = createEmptyMatrixLike act.cell_state_updated; error_forget = createEmptyMatrixLike act.activation_forget; error_input = createEmptyMatrixLike act.activation_input; error_block = createEmptyMatrixLike act.activation_block}

let lstm_error_cell right left_activations cur_activations weights (cur_errors: lstmErrors) =

    //printfn "error_block_output=%f" (deb.Apply(cur_errors.error_block_output))

    let error_output = 
        errorOutputModule.Apply(cur_errors.error_block_output,cur_activations.cell_state_updated,cur_activations.activation_output,cur_errors.error_output)

    //printfn "error_output=%f" (deb.Apply(error_output))

    let error_cell_state = 
        errorCellStateModule.Apply(cur_errors.error_block_output,cur_activations.activation_output,cur_activations.cell_state_updated,cur_errors.error_cell_state)

    broadcastingModule.BroadcastMultiply(weights.weights_peephole_output,error_output, cur_errors.error_cell_state) |> ignore

    match right with
        | Some (right_activations, right_errors) -> 
            broadcastingModule.BroadcastMultiply(weights.weights_peephole_input, right_errors.error_input, error_cell_state) |> ignore
            broadcastingModule.BroadcastMultiply(weights.weights_peephole_forget, right_errors.error_forget, error_cell_state) |> ignore
            elementwiseMultiplicationAndAdditionModule.Apply(right_errors.error_cell_state, right_activations.activation_forget, error_cell_state, error_cell_state) |> ignore
        | None -> ()

    //printfn "error_output=%f" (deb.Apply(error_cell_state))

    let error_forget = 
        match left_activations with
            | Some left_activations ->
                errorForgetModule.Apply(error_cell_state,left_activations.cell_state_updated,cur_activations.activation_forget,cur_errors.error_forget)
            | None ->
                setModule.Apply(0.0f,cur_errors.error_forget,cur_errors.error_forget)

    let errorInputModule = errorForgetModule
    let error_input = 
        errorInputModule.Apply(error_cell_state,cur_activations.activation_block,cur_activations.activation_input,cur_errors.error_input)

    // It just so happens that input block and the output have the same tanh activation.
    let errorBlockModule = errorCellStateModule

    let error_block = 
        errorBlockModule.Apply(error_cell_state,cur_activations.activation_input,cur_activations.activation_block,cur_errors.error_block)
    ()

// For the top layer
let lstm_error_top_layer target output (right: (lstmActivations*lstmErrors) option) left_activations cur_activations weights cur_errors =
    binaryErrorModule.Apply(target,output,cur_errors.error_block_output) |> ignore

    match right with
        | Some (_,er_r) ->
            sgemm2 nT nT 1.0f weights.weights_hidden_block er_r.error_block 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_input er_r.error_input 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_forget er_r.error_forget 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_output er_r.error_output 1.0f cur_errors.error_block_output |> ignore
        | None -> ()

    lstm_error_cell right left_activations cur_activations weights cur_errors

let lstm_error_middle_layer up (right: (lstmActivations*lstmErrors) option) left_activations cur_activations weights cur_errors =
    let er_flag =
        match up with
            | Some (up_pars, er_up) -> 
                sgemm2 nT nT 1.0f up_pars.weights_input_block er_up.error_block 0.0f cur_errors.error_block_output |> ignore
                sgemm2 nT nT 1.0f up_pars.weights_input_input er_up.error_input 1.0f cur_errors.error_block_output |> ignore
                sgemm2 nT nT 1.0f up_pars.weights_input_forget er_up.error_forget 1.0f cur_errors.error_block_output |> ignore
                sgemm2 nT nT 1.0f up_pars.weights_input_output er_up.error_output 1.0f cur_errors.error_block_output |> ignore
                1.0f
            | None -> 0.0f

    match right with
        | Some (_,er_r) ->
            sgemm2 nT nT 1.0f weights.weights_hidden_block er_r.error_block er_flag cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_input er_r.error_input 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_forget er_r.error_forget 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_output er_r.error_output 1.0f cur_errors.error_block_output |> ignore
        | None -> ()

    lstm_error_cell right left_activations cur_activations weights cur_errors

/// cur_errors.block_output does not get set to zero in this function, as it is expected that
/// rnn_backward_error_middle will set it to the starting value.
let lstm_error_feedforward_layer (right: (lstmActivations*lstmErrors) option) left_activations cur_activations weights cur_errors =
    match right with
        | Some (_,er_r) ->
            sgemm2 nT nT 1.0f weights.weights_hidden_block er_r.error_block 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_input er_r.error_input 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_forget er_r.error_forget 1.0f cur_errors.error_block_output |> ignore
            sgemm2 nT nT 1.0f weights.weights_hidden_output er_r.error_output 1.0f cur_errors.error_block_output |> ignore
        | None -> ()

    lstm_error_cell right left_activations cur_activations weights cur_errors

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

let createGradsLikeLSTM (w: lstmPars) = {
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

let addGradsToWeightsLSTM lstm_weights lstm_grads =
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

    sgeam2 nT nT 1.0f w.weights_peephole_input 1.0f g.grad_weights_peephole_input w.weights_peephole_input |> ignore
    sgeam2 nT nT 1.0f w.weights_peephole_forget 1.0f g.grad_weights_peephole_forget w.weights_peephole_forget |> ignore
    sgeam2 nT nT 1.0f w.weights_peephole_output 1.0f g.grad_weights_peephole_output w.weights_peephole_output |> ignore

    sgeam2 nT nT 1.0f w.bias_hidden_block 1.0f g.grad_bias_hidden_block w.bias_hidden_block |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_input 1.0f g.grad_bias_hidden_input w.bias_hidden_input |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_forget 1.0f g.grad_bias_hidden_forget w.bias_hidden_forget |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_output 1.0f g.grad_bias_hidden_output w.bias_hidden_output |> ignore

let applyGradientClippingLSTM lstm_grads coef =
    let g = lstm_grads

    gradclipModule.Apply(coef,g.grad_weights_input_block,g.grad_weights_input_block) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_input_input,g.grad_weights_input_input) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_input_forget,g.grad_weights_input_forget) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_input_output,g.grad_weights_input_output) |> ignore

    gradclipModule.Apply(coef,g.grad_weights_hidden_block,g.grad_weights_hidden_block) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_hidden_input,g.grad_weights_hidden_input) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_hidden_forget,g.grad_weights_hidden_forget) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_hidden_output,g.grad_weights_hidden_output) |> ignore

    gradclipModule.Apply(coef,g.grad_weights_peephole_input,g.grad_weights_peephole_input) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_peephole_forget,g.grad_weights_peephole_forget) |> ignore
    gradclipModule.Apply(coef,g.grad_weights_peephole_output,g.grad_weights_peephole_output) |> ignore

    gradclipModule.Apply(coef,g.grad_bias_hidden_block,g.grad_bias_hidden_block) |> ignore
    gradclipModule.Apply(coef,g.grad_bias_hidden_input,g.grad_bias_hidden_input) |> ignore
    gradclipModule.Apply(coef,g.grad_bias_hidden_forget,g.grad_bias_hidden_forget) |> ignore
    gradclipModule.Apply(coef,g.grad_bias_hidden_output,g.grad_bias_hidden_output) |> ignore

// For debugging.
let sumGradsLSTM lstm_grads =
    let g = lstm_grads

    printfn "g.grad_weights_input_block=%f" (deb.Apply(g.grad_weights_input_block))
    printfn "g.grad_weights_input_input=%f" (deb.Apply(g.grad_weights_input_input))
    printfn "g.grad_weights_input_forget=%f" (deb.Apply(g.grad_weights_input_forget))
    printfn "g.grad_weights_input_output=%f" (deb.Apply(g.grad_weights_input_output))

    printfn "g.grad_weights_hidden_block=%f" (deb.Apply(g.grad_weights_hidden_block))
    printfn "g.grad_weights_hidden_input=%f" (deb.Apply(g.grad_weights_hidden_input))
    printfn "g.grad_weights_hidden_forget=%f" (deb.Apply(g.grad_weights_hidden_forget))
    printfn "g.grad_weights_hidden_output=%f" (deb.Apply(g.grad_weights_hidden_output))

    printfn "g.grad_weights_peephole_input=%f" (deb.Apply(g.grad_weights_peephole_input))
    printfn "g.grad_weights_peephole_forget=%f" (deb.Apply(g.grad_weights_peephole_forget))
    printfn "g.grad_weights_peephole_output=%f" (deb.Apply(g.grad_weights_peephole_output))

    printfn "g.grad_bias_hidden_block=%f" (deb.Apply(g.grad_bias_hidden_block))
    printfn "g.grad_bias_hidden_input=%f" (deb.Apply(g.grad_bias_hidden_input))
    printfn "g.grad_bias_hidden_forget=%f" (deb.Apply(g.grad_bias_hidden_forget))
    printfn "g.grad_bias_hidden_output=%f" (deb.Apply(g.grad_bias_hidden_output))

let sumWeightsLSTM lstm_weights =
    let w = lstm_weights

    printfn "w.weights_input_block=%f" (deb.Apply(w.weights_input_block))
    printfn "w.weights_input_input=%f" (deb.Apply(w.weights_input_input))
    printfn "w.weights_input_forget=%f" (deb.Apply(w.weights_input_forget))
    printfn "w.weights_input_output=%f" (deb.Apply(w.weights_input_output))

    printfn "w.weights_hidden_block=%f" (deb.Apply(w.weights_hidden_block))
    printfn "w.weights_hidden_input=%f" (deb.Apply(w.weights_hidden_input))
    printfn "w.weights_hidden_forget=%f" (deb.Apply(w.weights_hidden_forget))
    printfn "w.weights_hidden_output=%f" (deb.Apply(w.weights_hidden_output))

    printfn "w.weights_peephole_input=%f" (deb.Apply(w.weights_peephole_input))
    printfn "w.weights_peephole_forget=%f" (deb.Apply(w.weights_peephole_forget))
    printfn "w.weights_peephole_output=%f" (deb.Apply(w.weights_peephole_output))

    printfn "w.bias_hidden_block=%f" (deb.Apply(w.bias_hidden_block))
    printfn "w.bias_hidden_input=%f" (deb.Apply(w.bias_hidden_input))
    printfn "w.bias_hidden_forget=%f" (deb.Apply(w.bias_hidden_forget))
    printfn "w.bias_hidden_output=%f" (deb.Apply(w.bias_hidden_output))

let sumErrors ers =
    printfn "ers.error_block_output=%f" (deb.Apply(ers.error_block_output))
    printfn "ers.error_output=%f" (deb.Apply(ers.error_output))
    printfn "ers.error_cell_state=%f" (deb.Apply(ers.error_cell_state))
    printfn "ers.error_forget=%f" (deb.Apply(ers.error_forget))
    printfn "ers.error_input=%f" (deb.Apply(ers.error_input))
    printfn "ers.error_block=%f" (deb.Apply(ers.error_block))
