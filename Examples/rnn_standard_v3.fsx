// The unfolded RNN that can be run for an arbitrary number of timesteps.

// My design of the backwards step was piss poor.
// The result is also consistently different from the
// other method. I am not sure whether this is an error.
// I am going to try breaking up the backwards step function.

// Edit: As it turns, the design of this script is bad.
// Because all the onsite allocation, it led to memory conflicts in the Reber grammar example.
// It will have to be remade.

// This here is that remake.

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

/// Computes the squared error of all the elements.
let squaredCostModule = new DeviceBinaryMapReduceModule <@ fun y a -> (y-a)*(y-a) @>

/// It is much better to clip the values in the final layer to [0.001,0.999] than to
/// fiddle with this function by clipping the value.
let crossEntropyCostModule = new DeviceBinaryMapReduceModule <@ fun a b -> -(a*(log b) + (1.0f-a)*log (1.0f - b)) @>

/// For errors without activations in the final layer such as when using the cross entropy cost.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

/// For errors in the middle layers using sparse activations. It also works for Relus
let binarySparseErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y c -> if c <> 0.0f then y else 0.0f @>

/// Relu activation. Max(0,x).
let reluActivationModule = new DeviceUnaryTransformModule<float32> <@ fun a -> if a > 0.0f then a else 0.0f @>

[<ReflectedDefinition>]
let inline sigmoid x = 1.0f / (1.0f+exp(-x))
[<ReflectedDefinition>]
let inline steep_sigmoid x = 1.0f / (1.0f+exp(-3.0f*x))

let logisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> sigmoid x @>

let steepLogisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> steep_sigmoid x @>

[<ReflectedDefinition>]
let inline clip x min_x max_x = (max (min x max_x) min_x)

[<ReflectedDefinition>]
let inline clipped_linear_sigmoid x =
    let t = clip x -0.5f 0.5f
    t+0.5f

let clippedLinearLogisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> clipped_linear_sigmoid x @>

[<ReflectedDefinition>]
let inline constrained_clipped_linear_sigmoid x =
    let t = clip x -0.499f 0.499f
    t+0.5f

[<ReflectedDefinition>]
let inline clipped_linear_tanh a = clip a -1.0f 1.0f

/// In the final layer having values too close to zero or one will cause explosive action in error derivatives.
/// This is a special sigmoid so that it does not happen. The values for it are clipped in the [0.001,0.999] range
let constrainedClippedLinearLogisticActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun x -> constrained_clipped_linear_sigmoid x @>

let tanhActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun a -> tanh a @>

let clippedLinearTanhActivationModule = 
    new DeviceUnaryTransformModule<float32> 
        <@ fun a -> clipped_linear_tanh a @>

[<ReflectedDefinition>]
let inline sigmoid_derivative c = c*(1.0f-c)
[<ReflectedDefinition>]
let inline clipped_linear_sigmoid_derivative c = if c <= 0.0f || c >= 1.0f then 0.0f else 1.0f
[<ReflectedDefinition>]
let inline constrained_clipped_linear_sigmoid_derivative c = if c <= 0.001f || c >= 0.999f then 0.0f else 1.0f
[<ReflectedDefinition>]
let inline clipped_linear_tanh_derivative c = if c <= -1.0f || c >= 1.0f then 0.0f else 1.0f 

/// For the sigmoid derivative
let logisticErrorModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> a * sigmoid_derivative b @>
let steepLogisticErrorModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> 3.0f * a * sigmoid_derivative b @>
let clippedLinearLogisticErrorModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> a * clipped_linear_sigmoid_derivative b @>
let contrainedClippedLinearLogisticErrorModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> a * constrained_clipped_linear_sigmoid_derivative b @>
let clippedLinearTanhErrorModule = new DeviceBinaryTransformModule<float32> <@ fun a b -> a * clipped_linear_tanh_derivative b @>

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

    momentum_weights_input_hidden : dM
    momentum_weights_hidden_hidden : dM option
    momentum_bias_hidden : dM

    momentum_flag_input : float32 ref
    momentum_flag_hidden : float32 ref
    momentum_flag_bias : float32 ref

    /// For Nesterov's Momentum
    weights_input_hidden_copy : dM
    weights_hidden_hidden_copy : dM option
    bias_hidden_copy : dM
    }

let createWeightPars xh hh bias_h = {
    weights_input_hidden = xh
    weights_hidden_hidden = hh
    bias_hidden = bias_h

    momentum_weights_input_hidden = createEmptyAndSetZero xh
    momentum_weights_hidden_hidden = createEmptyAndSetZero2 hh
    momentum_bias_hidden = createEmptyAndSetZero bias_h

    momentum_flag_input = ref 0.0f
    momentum_flag_hidden = ref 0.0f
    momentum_flag_bias = ref 0.0f

    weights_input_hidden_copy = sgeam nT nT 1.0f xh 0.0f xh
    weights_hidden_hidden_copy = dynamic_add nT nT 1.0f hh 0.0f hh None
    bias_hidden_copy = sgeam nT nT 1.0f bias_h 0.0f bias_h
    }

open System.Collections.Generic
let optional_get (dict: Dictionary<'a,'b>) key =
    if dict.ContainsKey(key) then Some dict.[key] else None

let rnn_forward (pars_dict: Dictionary<int,weightPars>) forward_dict row col (activation_module: DeviceUnaryTransformModule<float32>) =
    let prev_state = optional_get forward_dict (row,col-1)
    let input = optional_get forward_dict (row-1,col)
    let cur_act_start = optional_get forward_dict (row, col)

    let pars = pars_dict.[row]
    let multiply_flag = ref 0.0f
    
    let cur_act = dynamic_multiply nT nT 1.0f (Some pars.weights_input_hidden) input multiply_flag cur_act_start
    let cur_act = dynamic_multiply nT nT 1.0f pars.weights_hidden_hidden prev_state multiply_flag cur_act

    match pars.weights_hidden_hidden, prev_state with
        | None, Some x -> failwith "No hidden weights!"
        | _ -> ()

    if !multiply_flag = 0.0f then failwith "No operations done in forward step!"
    let cur_act = cur_act.Value
    addBias cur_act pars.bias_hidden
    let cur_act = activation_module.Apply(cur_act,cur_act)

    match cur_act_start with
        | None -> forward_dict.Add((row,col),cur_act)
        | Some x -> ()

let rnn_error_label (label_dict: Dictionary<int*int,dM>) (forward_dict: Dictionary<int*int,dM>) (error_dict: Dictionary<int*int,dM>) =
    for x in label_dict do
        let k = x.Key
        let target = x.Value
        let output = forward_dict.[k]

        let er_start = optional_get error_dict k
        match er_start with
            | Some er -> binaryErrorModule.Apply(target,output,er) |> ignore
            | None ->
                let t = binaryErrorModule.Apply(target,output)
                error_dict.Add(k,t)

let rnn_error (pars_dict: Dictionary<int,weightPars>) (forward_dict: Dictionary<int*int,dM>) (error_dict: Dictionary<int*int,dM>) row col (errorModule: DeviceBinaryTransformModule<float32>) =
    let cur_act = forward_dict.[row,col]
    let er_up = optional_get error_dict (row+1,col)    
    let er_right = optional_get error_dict (row,col+1)    
    let er_start = optional_get error_dict (row,col)
    
    let er_flag = ref 0.0f

    let weights_up = pars_dict.[row+1].weights_input_hidden
    let weights_hidden = pars_dict.[row].weights_hidden_hidden

    let cur_er = dynamic_multiply T nT 1.0f (Some weights_up) er_up er_flag er_start
    let cur_er = dynamic_multiply T nT 1.0f weights_hidden er_right er_flag cur_er
    let cur_er = errorModule.Apply(cur_er.Value, cur_act, cur_er.Value)

    match er_start with
        | None -> error_dict.Add((row,col),cur_er)
        | Some x -> ()

let rnn_set_momentum_flags (pars_dict: Dictionary<int,weightPars>) momentum_rate =
    for x in pars_dict do
        x.Value.momentum_flag_input := momentum_rate
        x.Value.momentum_flag_hidden := momentum_rate
        x.Value.momentum_flag_bias := momentum_rate

let rnn_gradient_calculate (pars_dict: Dictionary<int,weightPars>) (forward_dict: Dictionary<int*int,dM>) (error_dict: Dictionary<int*int,dM>) learning_coef row col =
    let act_left = optional_get forward_dict (row,col-1)    
    let act_down = optional_get forward_dict (row-1,col)

    let er_cur = error_dict.[row,col]
    let pars_cur = pars_dict.[row]
        
    let t = dynamic_multiply nT T learning_coef (Some er_cur) act_left pars_cur.momentum_flag_hidden pars_cur.momentum_weights_hidden_hidden
    let t = dynamic_multiply nT T learning_coef (Some er_cur) act_down pars_cur.momentum_flag_input (Some pars_cur.momentum_weights_input_hidden)
    dynamicCalculateBias learning_coef er_cur pars_cur.momentum_flag_bias pars_cur.momentum_bias_hidden
    ()

let rnn_gradient_add_to_weights (pars_dict: Dictionary<int,weightPars>) =
    for x in pars_dict do
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_weights_input_hidden) 1.0f (Some x.Value.weights_input_hidden) (Some x.Value.weights_input_hidden)
        let t = dynamic_add nT nT 1.0f x.Value.momentum_weights_hidden_hidden 1.0f x.Value.weights_hidden_hidden x.Value.weights_hidden_hidden
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_bias_hidden) 1.0f (Some x.Value.bias_hidden) (Some x.Value.bias_hidden)
        ()

/// Adds the momentum to the copy matrices. Used in Nesterov's Momentum.
let rnn_gradient_add_to_weights_nestorov (pars_dict: Dictionary<int,weightPars>) =
    for x in pars_dict do
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_weights_input_hidden) 1.0f (Some x.Value.weights_input_hidden_copy) (Some x.Value.weights_input_hidden_copy)
        let t = dynamic_add nT nT 1.0f x.Value.momentum_weights_hidden_hidden 1.0f x.Value.weights_hidden_hidden_copy x.Value.weights_hidden_hidden_copy
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_bias_hidden) 1.0f (Some x.Value.bias_hidden_copy) (Some x.Value.bias_hidden_copy)
        ()

let rnn_overwrite_with_copies_and_add_momentum (pars_dict: Dictionary<int,weightPars>) =
    for x in pars_dict do
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_weights_input_hidden) 1.0f (Some x.Value.weights_input_hidden_copy) (Some x.Value.weights_input_hidden)
        let t = dynamic_add nT nT 1.0f x.Value.momentum_weights_hidden_hidden 1.0f x.Value.weights_hidden_hidden_copy x.Value.weights_hidden_hidden
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_bias_hidden) 1.0f (Some x.Value.bias_hidden_copy) (Some x.Value.bias_hidden)
        ()

let elementwiseMultiplicationModule = 
    new DeviceBinaryTransformModule<float32> 
        <@ fun a b ->  
        a*b @>

let elementwiseMultiplicationAndAdditionModule = 
    new DeviceTrinaryTransformModule<float32> 
        <@ fun a b c ->  
        a*b+c @>

let gruOutputModule = 
    new DeviceQuadraryTransformModule<float32> 
        <@ fun a b c d->  
        a*b+c*d @>