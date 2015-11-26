// I've finally figured out where the problem is. It turns out I did not have a mechanism for using past cell states.
// Damn. At any rate I am going to have to repeat the experiment. This is also a good time to clean up some of the code.
// All those duplicate functions are dangerous.

// Library file for the LSTM functions.
// As the LSTM is quite large, it makes sense to pull out the components so I can call them from elsewhere.

//#load "utils.fsx"
#load "rnn_standard.fsx"
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

[<ReflectedDefinition>]
let inline sigmoid x = 1.0f / (1.0f+exp(-x))

let logisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> sigmoid x @>

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

let tanhActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun a -> tanh a @>
    
/// Clipped linear tanh
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

            //broadcastingModule.BroadcastMultiply(weights_peephole,cell_state,function_output) |> ignore
            //This was the big error. If the previous step does not have a hidden state, then it cannot have a cell state.

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

let lstm_error_cell error_block_output (right: (lstmActivations*lstmErrors) option) left_activations cur_activations weights cur_errors =
    let cur_error_block, cur_error_input, cur_error_forget, cur_error_cell_state, cur_error_output, cur_error_block_output =
        match cur_errors with
            | Some x -> (Some x.error_block), (Some x.error_input), (Some x.error_forget), (Some x.error_cell_state), (Some x.error_output), (Some x.error_block_output)
            | None -> None, None, None, None, None, None

    let error_block_output =
        match error_block_output, right with
            | Some error_block_output, Some (_, er_r) ->
                sgemm2 nT nT 1.0f weights.weights_hidden_block er_r.error_block 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f weights.weights_hidden_input er_r.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f weights.weights_hidden_forget er_r.error_forget 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f weights.weights_hidden_output er_r.error_output 1.0f error_block_output
            | None, Some (_, er_r) -> 
                let error_block_output = dynamic_multiply nT nT 1.0f weights.weights_hidden_block er_r.error_block 0.0f cur_error_block_output
                sgemm2 nT nT 1.0f weights.weights_hidden_input er_r.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f weights.weights_hidden_forget er_r.error_forget 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f weights.weights_hidden_output er_r.error_output 1.0f error_block_output
            | Some error_block_output, None -> error_block_output
            | None, None -> failwith "Invalid input to lstm_error_allocation_cell"

    let error_output = 
        match cur_error_output with
            | Some cur_error_output ->
                errorOutputModule.Apply(error_block_output,cur_activations.cell_state_updated,cur_activations.activation_output,cur_error_output)
            | None ->
                errorOutputModule.Apply(error_block_output,cur_activations.cell_state_updated,cur_activations.activation_output)

    let error_cell_state = 
        match cur_error_cell_state with
            | Some cur_error_cell_state ->
                errorCellStateModule.Apply(error_block_output,cur_activations.activation_output,cur_activations.cell_state_updated,cur_error_cell_state)
            | None -> 
                errorCellStateModule.Apply(error_block_output,cur_activations.activation_output,cur_activations.cell_state_updated)

    broadcastingModule.BroadcastMultiply(weights.weights_peephole_output,error_output, error_cell_state) |> ignore

    match right with
        | Some (right_activations, right_errors) -> 
            broadcastingModule.BroadcastMultiply(weights.weights_peephole_input, right_errors.error_input, error_cell_state) |> ignore
            broadcastingModule.BroadcastMultiply(weights.weights_peephole_forget, right_errors.error_forget, error_cell_state) |> ignore
            elementwiseMultiplicationAndAdditionModule.Apply(right_errors.error_cell_state, right_activations.activation_forget, error_cell_state, error_cell_state) |> ignore
        | None -> ()

    let error_forget = 
        match left_activations, cur_error_forget with
            | Some left_activations, Some cur_error_forget ->
                errorForgetModule.Apply(error_cell_state,left_activations.cell_state_updated,cur_activations.activation_forget,cur_error_forget)
            | Some left_activations, None ->
                errorForgetModule.Apply(error_cell_state,left_activations.cell_state_updated,cur_activations.activation_forget)
            | None, Some cur_error_forget ->
                setModule.Apply(0.0f,cur_error_forget,cur_error_forget)
            | None, None ->
                setModule.Apply(0.0f,error_cell_state)

    let errorInputModule = errorForgetModule
    let error_input = 
        match cur_error_input with
            | Some cur_error_input ->
                errorInputModule.Apply(error_cell_state,cur_activations.activation_block,cur_activations.activation_input,cur_error_input)
            | None ->
                errorInputModule.Apply(error_cell_state,cur_activations.activation_block,cur_activations.activation_input)

    // It just so happens that input block and the output have the same tanh activation.
    let errorBlockModule = errorCellStateModule

    let error_block = 
        match cur_error_block with
            | Some cur_error_block ->
                errorBlockModule.Apply(error_cell_state,cur_activations.activation_input,cur_activations.activation_block,cur_error_block)
            | None ->
                errorBlockModule.Apply(error_cell_state,cur_activations.activation_input,cur_activations.activation_block)
    {
    error_block_output = error_block_output
    error_output = error_output
    error_cell_state = error_cell_state
    error_forget = error_forget
    error_input = error_input
    error_block = error_block
    }

// For the top layer
let lstm_error_top_layer target output (right: (lstmActivations*lstmErrors) option) left_activations cur_activations weights cur_errors =
    let error_block_output = 
        match cur_errors with
            | Some cur_errors ->
                binaryErrorModule.Apply(target,output,cur_errors.error_block_output)
            | None ->
                binaryErrorModule.Apply(target,output)
    lstm_error_cell (Some error_block_output) right left_activations cur_activations weights cur_errors

let lstm_error_middle_layer up (right: (lstmActivations*lstmErrors) option) left_activations cur_activations weights cur_errors =
    let cur_error_block_output =
        match cur_errors with
            | Some cur_errors -> Some cur_errors.error_block_output
            | None -> None

    let error_block_output =
        match up with
            | Some (up_pars, er_up) -> 
                let error_block_output = dynamic_multiply nT nT 1.0f up_pars.weights_input_block er_up.error_block 0.0f cur_error_block_output
                sgemm2 nT nT 1.0f up_pars.weights_input_input er_up.error_input 1.0f error_block_output |> ignore
                sgemm2 nT nT 1.0f up_pars.weights_input_forget er_up.error_forget 1.0f error_block_output |> ignore
                Some (sgemm2 nT nT 1.0f up_pars.weights_input_output er_up.error_output 1.0f error_block_output)
            | None -> None

    lstm_error_cell error_block_output right left_activations cur_activations weights cur_errors

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

    sgeam2 nT nT 1.0f w.weights_peephole_output 1.0f g.grad_weights_peephole_output w.weights_peephole_output |> ignore
    sgeam2 nT nT 1.0f w.weights_peephole_output 1.0f g.grad_weights_peephole_output w.weights_peephole_output |> ignore
    sgeam2 nT nT 1.0f w.weights_peephole_output 1.0f g.grad_weights_peephole_output w.weights_peephole_output |> ignore

    sgeam2 nT nT 1.0f w.bias_hidden_block 1.0f g.grad_bias_hidden_block w.bias_hidden_block |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_input 1.0f g.grad_bias_hidden_input w.bias_hidden_input |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_forget 1.0f g.grad_bias_hidden_forget w.bias_hidden_forget |> ignore
    sgeam2 nT nT 1.0f w.bias_hidden_output 1.0f g.grad_bias_hidden_output w.bias_hidden_output |> ignore

let d_training_sequence1 = Some ({num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;1.0f;1.0f|])}:dM)
let d_training_sequence2 = Some ({num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;1.0f;0.0f;1.0f|])}:dM)
let d_target_sequence = {num_rows=4; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;0.0f;0.5f;0.0f;0.0f;0.5f;0.0f;0.0f;0.5f;0.0f;0.0f;0.5f;0.0f;0.0f;0.0f|])}:dM

let hidden_size = 50
let batch_size = d_training_sequence1.Value.num_cols

let l1 = createRandomLstmCell hidden_size 1
let l2 = createRandomLstmCell d_target_sequence.num_rows hidden_size

let a1 = lstm_activation l1 d_training_sequence1 None None
let a2 = lstm_activation l1 d_training_sequence2 (Some a1) None
let b2 = lstm_activation l2 (Some a2.block_output) None None

let er_b2 = lstm_error_top_layer d_target_sequence b2.block_output None None b2 l2 None
let er_a2 = lstm_error_middle_layer (Some (l2,er_b2)) None (Some a1) a2 l1 None
let er_a1 = lstm_error_middle_layer None (Some (a2,er_a2)) None a1 l1 None

let g1 = createGradsLikeLSTM l1
let g2 = createGradsLikeLSTM l2

let lstm_test num_iterations learning_coef momentum_rate =
    
    for i=1 to num_iterations do

        let a1 = lstm_activation l1 d_training_sequence1 None (Some a1)
        let a2 = lstm_activation l1 d_training_sequence2 (Some a1) (Some a2)
        let b2 = lstm_activation l2 (Some a2.block_output) None (Some b2)

        let sq_er = squaredCostModule.Apply(d_target_sequence,b2.block_output) * 0.25f
        printfn "Squared error cost is %f at iteration %i" sq_er i

        let er_b2 = lstm_error_top_layer d_target_sequence b2.block_output None None b2 l2 (Some er_b2)
        let er_a2 = lstm_error_middle_layer (Some (l2,er_b2)) None (Some a1) a2 l1 (Some er_a2)
        let er_a1 = lstm_error_middle_layer None (Some (a2,er_a2)) None a1 l1 (Some er_a1)

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

        addGradsToWeightsLSTM l1 g1
        addGradsToWeightsLSTM l2 g2

let inv_batch_size = d_training_sequence1.Value.num_cols |> float32
let learning_rate = 0.1f
let learning_coef = -inv_batch_size*learning_rate
let momentum_rate = 0.9f

lstm_test 500 learning_coef momentum_rate