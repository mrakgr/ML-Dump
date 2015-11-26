// The unfolded RNN that can be run for an arbitrary number of timesteps.

// My design of the backwards step was piss poor.
// The result is also consistently different from the
// other method. I am not sure whether this is an error.
// I am going to try breaking up the backwards step function.

// Edit: As it turns, the design of this script is bad.
// Because all the onsite allocation, it led to memory conflicts in the Reber grammar example.
// It will have to be remade.

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

[<ReflectedDefinition>]
let inline sigmoid x = 1.0f / (1.0f+exp(-x))

let logisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> sigmoid x @>

let tanhActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun a -> tanh a @>

[<ReflectedDefinition>]
let inline sigmoid_derivative c = c*(1.0f-c)

/// For the sigmoid derivative
let logisticErrorModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> a * sigmoid_derivative b @>

[<ReflectedDefinition>]
let inline tanh_derivative c = 1.0f-c*c

/// For the tanh derivative
let tanhErrorModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> a * tanh_derivative b @>

// Gradient clipping module.
let gradclipModule = 
    new DeviceUnaryCoefTransformModule<float32>
        <@ fun coef_a a ->
        if a > coef_a then coef_a
        else if a < -coef_a then -coef_a
        else a @>

type weightPars = {
    weights_input_hidden : dM
    weights_hidden_hidden : dM option
    bias_hidden : dM
    }

type gradPars = {
    grad_input_hidden : dM
    grad_hidden_hidden : dM option
    grad_bias_hidden : dM
    }

// Dynamically allocates memory if the matrix has not been used before.
let dynamic_multiply T1 T2 alpha weights input beta dest =
    match dest with
        | Some dest -> sgemm2 T1 T2 alpha weights input beta dest
        | None -> sgemm T1 T2 alpha weights input

let rnn_forward (p : weightPars) (input: dM option) (prev_hidden: dM option) (cur_hidden: dM option) (activationModule: DeviceUnaryTransformModule<float32>) = 
    let weights_input_hidden : dM = p.weights_input_hidden
    let weights_hidden_hidden = p.weights_hidden_hidden
    let bias_hidden : dM = p.bias_hidden

    // Forward pass.
    match prev_hidden, input with
        | Some prev, Some inp ->
            let cur_hidden = dynamic_multiply T nT 1.0f weights_input_hidden inp 0.0f cur_hidden
            
            match weights_hidden_hidden with
                | Some weights_hidden_hidden ->
                    sgemm2 T nT 1.0f weights_hidden_hidden prev 1.0f cur_hidden |> ignore
                | None -> failwith "State from the left layer cannot go into the top layer which has no hidden weights."
            
            addBias cur_hidden bias_hidden
            activationModule.Apply(cur_hidden,cur_hidden)
        | Some prev, None ->
            match weights_hidden_hidden with
                | Some weights_hidden_hidden ->
                    let cur_hidden = dynamic_multiply T nT 1.0f weights_hidden_hidden prev 0.0f cur_hidden
                    addBias cur_hidden bias_hidden
                    activationModule.Apply(cur_hidden,cur_hidden)
                | None -> failwith "State from the left layer cannot go into the top layer which has no hidden weights."
        | None, Some inp ->
            let cur_hidden = dynamic_multiply T nT 1.0f weights_input_hidden inp 0.0f cur_hidden
            addBias cur_hidden bias_hidden
            activationModule.Apply(cur_hidden,cur_hidden)
        | None, None -> failwith "Invalid input. A forward step must have either prev_hidden or input or both."

// Computes the derivatives with respect to the top layer.
let rnn_backward_error_top (target: dM) activations_cur (er_cur: dM option)  = 
    match er_cur with
        | Some out ->
            binaryErrorModule.Apply(target,activations_cur,out) |> ignore
            out
        | None -> 
            binaryErrorModule.Apply(target,activations_cur)

// Computes the derivatives with respect to the middle layers.
let rnn_backward_error_middle (er_p_above: (dM*weightPars) option) (er_p_right: (dM*weightPars) option) (er_cur: dM option) (activations_cur: dM) (errorModule: DeviceBinaryTransformModule<float32>) = 
    let er_flag, er_cur = 
        match er_p_above with
            | Some (er_above, p_above) ->
                let weights_hidden_output : dM = p_above.weights_input_hidden
            
                let er_cur = dynamic_multiply nT nT 1.0f weights_hidden_output er_above 0.0f er_cur
                1.0f, Some er_cur
            | None -> 0.0f, er_cur

    let er_cur =
        match er_p_right with
            | Some (er_right, p) ->
                let weights_hidden_hidden : dM = 
                    match p.weights_hidden_hidden with
                        | Some x -> x
                        | None -> failwith "Function can only be used on the middle layers where hidden to hidden connection exist."

                let er_cur = dynamic_multiply nT nT 1.0f weights_hidden_hidden er_right er_flag er_cur
                Some er_cur
            | None -> er_cur

    // Applies the derivative of the activation function.
    // Will throw an exception if both er_above and er_right are None.
    errorModule.Apply(er_cur.Value,activations_cur, er_cur.Value) |> ignore
    er_cur.Value

// Calculates the weight gradients.
let rnn_backwards_weight (er_p_g_b_above: (dM*weightPars*gradPars*bool) option) (er_p_g_b_right: (dM*weightPars*gradPars*bool) option) (activations_cur: dM) learning_coef momentum_rate = 
    match er_p_g_b_above with
        | Some (er_above, p_above, g_above, do_calculate_bias) ->
            let bias_out : dM = p_above.bias_hidden
            let weights_hidden_output : dM = p_above.weights_input_hidden
            let grad_bias_out: dM = g_above.grad_bias_hidden
            let grad_hidden_output: dM = g_above.grad_input_hidden

            sgemm2 nT T learning_coef activations_cur er_above momentum_rate grad_hidden_output |> ignore
            if do_calculate_bias then calculateBias learning_coef er_above momentum_rate grad_bias_out
        | None -> ()

    match er_p_g_b_right with
        | Some (er_right, p, g, do_calculate_bias) ->
            let bias_hidden : dM = p.bias_hidden
            let weights_input_hidden : dM = p.weights_input_hidden
            let weights_hidden_hidden : dM = 
                match p.weights_hidden_hidden with
                    | Some x -> x
                    | None -> failwith "Function can only be used on the middle layers where hidden to hidden connection exist."

            let grad_bias_hidden: dM = g.grad_bias_hidden
            let grad_input_hidden: dM = g.grad_input_hidden
            let grad_hidden_hidden: dM = 
                match g.grad_hidden_hidden with
                    | Some x -> x
                    | None -> failwith "Function can only be used on the middle layers where hidden to hidden connection exist."

            sgemm2 nT T learning_coef activations_cur er_right momentum_rate grad_hidden_hidden |> ignore
            if do_calculate_bias then calculateBias learning_coef er_right momentum_rate grad_bias_hidden
        | None -> ()

let createEmptyAndSetZero (example: dM) =
    let t = createEmptyMatrixLike example
    setModule.Apply(0.0f, t, t)

let createEmptyAndSetZero2 (example: dM option) =
    match example with
        | Some example ->
            let t = createEmptyMatrixLike example
            Some (setModule.Apply(0.0f, t, t))
        | None -> None

let createRandomMatrix a b =
    let scale = 1.0f/sqrt((a+b) |> float32)
    let location = -scale*0.5f
    createRandomUniformMatrix a b scale location

let createRandomFeedforwardWeights input_size output_size = {
    weights_input_hidden = createRandomMatrix input_size output_size
    weights_hidden_hidden = None

    bias_hidden = createRandomMatrix output_size 1
    }

let createRandomRecurrentWeights input_size output_size = {
    weights_input_hidden = createRandomMatrix input_size output_size 
    weights_hidden_hidden = Some (createRandomMatrix output_size output_size )

    bias_hidden = createRandomMatrix output_size 1
    }

let createGradsLike l = {
    grad_input_hidden = createEmptyAndSetZero l.weights_input_hidden
    grad_hidden_hidden = createEmptyAndSetZero2 l.weights_hidden_hidden
    grad_bias_hidden = createEmptyAndSetZero l.bias_hidden
    }

// Adds the gradients to the weights. For Nesterov's momentum.
let addGradsToWeights momentum_rate grad_pars weight_pars copy_pars =
    sgeam2 nT nT momentum_rate grad_pars.grad_bias_hidden 1.0f weight_pars.bias_hidden copy_pars.bias_hidden |> ignore
    sgeam2 nT nT momentum_rate grad_pars.grad_input_hidden 1.0f weight_pars.weights_input_hidden copy_pars.weights_input_hidden |> ignore

    match grad_pars.grad_hidden_hidden, weight_pars.weights_hidden_hidden, copy_pars.weights_hidden_hidden with
        | Some grad_hidden_hidden, Some weights_hidden_hidden, Some copy_weights_hidden_hidden->
            sgeam2 nT nT momentum_rate grad_hidden_hidden 1.0f weights_hidden_hidden copy_weights_hidden_hidden |> ignore
        | None, None, None -> ()
        | _ -> failwith "Invalid inputs to addGradsToWeights"

// Adds the gradients to the weights. For Nesterov's momentum.
let applyGradientClipping grad_pars coef =
    gradclipModule.Apply(coef,grad_pars.grad_bias_hidden,grad_pars.grad_bias_hidden) |> ignore
    gradclipModule.Apply(coef,grad_pars.grad_input_hidden,grad_pars.grad_input_hidden) |> ignore

    match grad_pars.grad_hidden_hidden with
        | Some grad_hidden_hidden ->
            gradclipModule.Apply(coef,grad_hidden_hidden,grad_hidden_hidden) |> ignore
        | None -> ()


// Debugging stuff.
let deb = new DeviceUnaryMapReduceModule <@ fun a -> abs(a) @>

let sumGrads grad_pars =
    printfn "grad_pars.grad_bias_hidden=%f" (deb.Apply(grad_pars.grad_bias_hidden))
    printfn "grad_pars.grad_input_hidden=%f" (deb.Apply(grad_pars.grad_input_hidden) )

    match grad_pars.grad_hidden_hidden with
        | Some grad_hidden_hidden ->
            printfn "grad_hidden_hidden=%f" (deb.Apply(grad_hidden_hidden))
        | None -> ()

let sumWeights weight_pars =
    printfn "weight_pars.grad_bias_hidden=%f" (deb.Apply(weight_pars.bias_hidden))
    printfn "weight_pars.grad_input_hidden=%f" (deb.Apply(weight_pars.weights_input_hidden) )

    match weight_pars.weights_hidden_hidden with
        | Some weights_hidden_hidden ->
            printfn "weights_hidden_hidden=%f" (deb.Apply(weights_hidden_hidden))
        | None -> ()