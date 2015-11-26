// gradient_step_backward weight_pars weight_pars2 grad_pars grad_pars2 None None (Some a2) a1 d_training_sequence1 learning_coef 1.0f
// I spent six hours trying to figure out why the net blows up before I noticed that the hidden layers have 1.0f momentum like in
// the line above.
// Fuck.
// The biases are not calculated correctly either.
// I'll dump this here for now just incase I need the debug code later.

// 3h later: Well, speak of the devil.

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

// Gradient clipping module.
let gradclipModule = 
    new DeviceUnaryCoefTransformModule<float32>
        <@ fun coef_a a ->
        if a > coef_a then coef_a
        else if a < -coef_a then -coef_a
        else a@>

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

let gradient_step_forward (p : weightPars) (input: dM option) (prev_hidden: dM option) (cur_hidden: dM option) = 
    let weights_input_hidden : dM = p.weights_input_hidden
    let weights_hidden_hidden = p.weights_hidden_hidden
    let bias_hidden : dM = p.bias_hidden

    // Forward pass.
    // Dynamically allocates memory if the matrix has not been used before.
    let dynamic_multiply T1 T2 alpha weights input beta dest =
        match dest with
            | Some dest -> sgemm2 T1 T2 alpha weights input beta dest
            | None -> sgemm T1 T2 alpha weights input

    match prev_hidden, input with
        | Some prev, Some inp ->
            let cur_hidden = dynamic_multiply T nT 1.0f weights_input_hidden inp 0.0f cur_hidden
            
            match weights_hidden_hidden with
                | Some weights_hidden_hidden ->
                    sgemm2 T nT 1.0f weights_hidden_hidden prev 1.0f cur_hidden |> ignore
                | None -> failwith "State from the left layer cannot go into the top layer which has no hidden weights."
            
            addBias cur_hidden bias_hidden
            reluActivationModule.Apply(cur_hidden,cur_hidden)
        | Some prev, None ->
            match weights_hidden_hidden with
                | Some weights_hidden_hidden ->
                    let cur_hidden = dynamic_multiply T nT 1.0f weights_hidden_hidden prev 0.0f cur_hidden
                    addBias cur_hidden bias_hidden
                    reluActivationModule.Apply(cur_hidden,cur_hidden)
                | None -> failwith "State from the left layer cannot go into the top layer which has no hidden weights."
        | None, Some inp ->
            let cur_hidden = dynamic_multiply T nT 1.0f weights_input_hidden inp 0.0f cur_hidden
            addBias cur_hidden bias_hidden
            reluActivationModule.Apply(cur_hidden,cur_hidden)
        | None, None -> failwith "Invalid input. A forward step must have either prev_hidden or input or both."

let gradient_step_backward p p_above (g : gradPars) g_above (er_above: dM option) (target_output: (dM*dM) option) (er_right: dM option) (er_cur: dM) (input: dM option) learning_coef momentum_rate = 
    let bias_hidden : dM = p.bias_hidden
    let bias_out : dM = p_above.bias_hidden
    let weights_input_hidden : dM = p.weights_input_hidden
    let weights_hidden_hidden : dM = 
        match p.weights_hidden_hidden with
            | Some x -> x
            | None -> failwith "Function can only be used on the middle layers where hidden to hidden connection exist."
    let weights_hidden_output : dM = p_above.weights_input_hidden

    let grad_bias_hidden: dM = g.grad_bias_hidden
    let grad_bias_out: dM = g_above.grad_bias_hidden
    let grad_input_hidden: dM = g.grad_input_hidden
    let grad_hidden_hidden: dM = 
        match g.grad_hidden_hidden with
            | Some x -> x
            | None -> failwith "Function can only be used on the middle layers where hidden to hidden connection exist."
    let grad_hidden_output: dM = g_above.grad_input_hidden

    // Backwards pass.
    // I repurposing the activation matrices in the hidden nodes for the error gradients.
    // I need to be careful to the calculations in the right order.
    // The code is a bit confusing as er_cur is used for both the error and the current activations.

    // Calculates the error from the upper layer.
    // er_above is for the middle layers and target_output is for the final layer.
    let er_out: dM option = 
        match er_above, target_output with
            | None, Some (target, out) ->
                binaryErrorModule.Apply(target,out,out) |> ignore
                Some out
            | Some er_out, None ->
                Some er_out
            | None, None -> None
            | Some x, Some y -> failwith "Invalid input in gradient_step_backward. There cannot be both er_above and output_target_er_out."

    // Calculates the gradient for the hidden to output weights, going from up to down.
    match er_out with
        | Some er_out ->
            sgemm2 nT T learning_coef er_cur er_out momentum_rate grad_hidden_output |> ignore
            calculateBias learning_coef er_out momentum_rate grad_bias_out
        | None -> ()

    // Calculates the gradient for the hidden to hidden weights, going from right to left.
    match er_right with
        | Some er_right ->
            sgemm2 nT T learning_coef er_cur er_right momentum_rate grad_hidden_hidden |> ignore
            calculateBias learning_coef er_right momentum_rate grad_bias_hidden
        | None -> ()

    // Calculates the error for the current node.
    match er_out, er_right with
        | Some er_out, Some er_right ->
            sgemm2 nT nT 1.0f weights_hidden_output er_out 0.0f er_cur |> ignore
            sgemm2 nT nT 1.0f weights_hidden_hidden er_right 1.0f er_cur |> ignore
        | Some er_out, None ->
            sgemm2 nT nT 1.0f weights_hidden_output er_out 0.0f er_cur |> ignore
        | None, Some er_right ->
            sgemm2 nT nT 1.0f weights_hidden_hidden er_right 0.0f er_cur |> ignore
        | None, None -> ()

    // Applies the derivative of the activation function.
    binarySparseErrorModule.Apply(er_cur,er_cur) |> ignore

    // For the first layer only.
    // Calculates the gradient with respect to the input to hidden weights.

    match input with
        | Some input -> sgemm2 nT T learning_coef input er_cur momentum_rate grad_input_hidden |> ignore
        | None -> ()

let hidden_size = 250

let weight_pars = {
    weights_input_hidden = createRandomUniformMatrix 1 hidden_size 0.1f -0.05f
    weights_hidden_hidden = Some (createRandomUniformMatrix hidden_size hidden_size 0.1f -0.05f)

    bias_hidden = createRandomUniformMatrix hidden_size 1 0.1f -0.05f
    }

let weight_pars2 = {
    weights_input_hidden = createRandomUniformMatrix hidden_size 4 0.1f -0.05f
    weights_hidden_hidden = None
    
    bias_hidden = createRandomUniformMatrix 4 1 0.1f -0.05f
    }

let createEmptyAndSetZero (example: dM) =
    let t = createEmptyMatrixLike example
    setModule.Apply(0.0f, t, t)

let createEmptyAndSetZero2 (example: dM option) =
    match example with
        | Some example ->
            let t = createEmptyMatrixLike example
            Some (setModule.Apply(0.0f, t, t))
        | None -> None

let grad_pars = {
    grad_input_hidden = createEmptyAndSetZero weight_pars.weights_input_hidden
    grad_hidden_hidden = createEmptyAndSetZero2 weight_pars.weights_hidden_hidden
    grad_bias_hidden = createEmptyAndSetZero weight_pars.bias_hidden
    }

let grad_pars2 = {
    grad_input_hidden = createEmptyAndSetZero weight_pars2.weights_input_hidden
    grad_hidden_hidden = createEmptyAndSetZero2 weight_pars2.weights_hidden_hidden
    grad_bias_hidden = createEmptyAndSetZero weight_pars2.bias_hidden
    }

let d_training_sequence1 = Some ({num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;1.0f;1.0f|])}:dM)
let d_training_sequence2 = Some ({num_rows=1; num_cols=4; dArray=worker.Malloc([|0.0f;1.0f;0.0f;1.0f|])}:dM)
let d_target_sequence = {num_rows=4; num_cols=4; dArray=worker.Malloc([|0.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;1.0f;0.0f;0.0f;0.0f|])}:dM

// Forward Steps
let a1 = gradient_step_forward weight_pars d_training_sequence1 None None
let a2 = gradient_step_forward weight_pars d_training_sequence2 (Some a1) None
let b2 = gradient_step_forward weight_pars2 (Some a2) None None


let prev_hidden_state = createEmptyAndSetZero a1

let p = weight_pars
let p_above = weight_pars2
let g = grad_pars
let g_above = grad_pars2

let bias_hidden : dM = p.bias_hidden
let bias_out : dM = p_above.bias_hidden
let weights_input_hidden : dM = p.weights_input_hidden
let weights_hidden_hidden : dM = 
    match p.weights_hidden_hidden with
        | Some x -> x
        | None -> failwith "Function can only be used on the middle layers where hidden to hidden connection exist."
let weights_hidden_output : dM = p_above.weights_input_hidden

let grad_bias_hidden: dM = g.grad_bias_hidden
let grad_bias_out: dM = g_above.grad_bias_hidden
let grad_input_hidden: dM = g.grad_input_hidden
let grad_hidden_hidden: dM = 
    match g.grad_hidden_hidden with
        | Some x -> x
        | None -> failwith "Function can only be used on the middle layers where hidden to hidden connection exist."
let grad_hidden_output: dM = g_above.grad_input_hidden

// Forward pass.
let xh = sgemm T nT 1.0f weights_input_hidden d_training_sequence1.Value
let hh = sgemm T nT 1.0f weights_hidden_hidden prev_hidden_state 

let combined_hidden = sgeam nT nT 1.0f xh 1.0f hh
addBias combined_hidden bias_hidden
reluActivationModule.Apply(combined_hidden,combined_hidden) |> ignore

let xh2 = sgemm T nT 1.0f weights_input_hidden d_training_sequence2.Value 
let hh2 = sgemm T nT 1.0f weights_hidden_hidden combined_hidden 

let combined_hidden2 = sgeam nT nT 1.0f xh2 1.0f hh2
addBias combined_hidden2 bias_hidden
reluActivationModule.Apply(combined_hidden2,combined_hidden2) |> ignore

let out = sgemm T nT 1.0f weights_hidden_output combined_hidden2
addBias out bias_out
reluActivationModule.Apply(out,out) |> ignore

let sq_er = squaredCostModule.Apply(d_target_sequence,b2) * 0.25f
printfn "Squared error cost is %f at iteration " sq_er

let sq_er2 = squaredCostModule.Apply(d_target_sequence,out) * 0.25f
printfn "Squared error cost is %f at iteration  using the original method" sq_er2 

// Backwards pass.

let er_out = binaryErrorModule.Apply(d_target_sequence,out)
        
let er_hidden2 = sgemm nT nT 1.0f weights_hidden_output er_out
binarySparseErrorModule.Apply(er_hidden2,er_hidden2) |> ignore

let er_hidden = sgemm nT nT 1.0f weights_hidden_hidden er_hidden2
binarySparseErrorModule.Apply(er_hidden,er_hidden) |> ignore

let target_out =
    Some (d_target_sequence, b2)

let inv_batch_size = 1.0f / float32 d_training_sequence1.Value.num_cols
let learning_rate = 0.1f
let learning_coef = (-inv_batch_size*learning_rate)
let momentum_rate = 0.9f

//let temp_w1 = (sgeam nT nT 1.0f weights_input_hidden 0.0f weights_input_hidden).dArray.Gather()
//let temp_w2 = (sgeam nT nT 1.0f weights_hidden_hidden 0.0f weights_hidden_hidden).dArray.Gather()
//let temp_w3 = (sgeam nT nT 1.0f weights_hidden_output 0.0f weights_hidden_output).dArray.Gather()
//let temp_w4 = (sgeam nT nT 1.0f bias_hidden 0.0f bias_hidden).dArray.Gather()
//let temp_w5 = (sgeam nT nT 1.0f bias_out 0.0f bias_out).dArray.Gather()

let test() =
    // Forward Steps
    let a1 = gradient_step_forward weight_pars d_training_sequence1 None None
    //let a2 = gradient_step_forward weight_pars d_training_sequence2 (Some a1) (Some a2)
    //let b2 = gradient_step_forward weight_pars2 (Some a2) None (Some b2)

    // Forward pass.
    let xh = sgemm T nT 1.0f weights_input_hidden d_training_sequence1.Value //0.0f xh
    let hh = sgemm T nT 1.0f weights_hidden_hidden prev_hidden_state //0.0f hh

    let combined_hidden = sgeam nT nT 1.0f xh 1.0f hh //combined_hidden
    addBias combined_hidden bias_hidden
    reluActivationModule.Apply(combined_hidden,combined_hidden) |> ignore

    let t1 = combined_hidden.dArray.Gather()
    let t2 = a1.dArray.Gather()
    let t = Array.map2(fun a b -> abs(a-b)) t1 t2
    printfn "%A" t

test()

// Backprop
gradient_step_backward weight_pars weight_pars2 grad_pars grad_pars2 None target_out None a2 d_training_sequence2 learning_coef momentum_rate
gradient_step_backward weight_pars weight_pars2 grad_pars grad_pars2 None None (Some a2) a1 d_training_sequence1 learning_coef 1.0f
//gradient_step_backward weight_pars weight_pars2 grad_pars grad_pars2 None None (Some a1) prev_hidden_state None learning_coef 1.0f

//let temp_w6 = (sgeam nT nT 1.0f weights_input_hidden 0.0f weights_input_hidden).dArray.Gather()
//let temp_w7 = (sgeam nT nT 1.0f weights_hidden_hidden 0.0f weights_hidden_hidden).dArray.Gather()
//let temp_w8 = (sgeam nT nT 1.0f weights_hidden_output 0.0f weights_hidden_output).dArray.Gather()
//let temp_w9 = (sgeam nT nT 1.0f bias_hidden 0.0f bias_hidden).dArray.Gather()
//let temp_w10 = (sgeam nT nT 1.0f bias_out 0.0f bias_out).dArray.Gather()

//let t1 = temp_w5
//let t2 = temp_w10
//let t = Array.map2(fun a b -> abs(a-b)) t1 t2
//t |> Array.sum




// Calculate gradients

let grad_hidden_output_t = sgemm nT T (-inv_batch_size*learning_rate) combined_hidden2 er_out
let grad_hidden_hidden_t = sgemm nT T (-inv_batch_size*learning_rate) combined_hidden er_hidden2
let grad_input_hidden_t = sgemm nT T (-inv_batch_size*learning_rate) d_training_sequence2.Value er_hidden2

//sgemm2 nT T (-inv_batch_size*learning_rate) prev_hidden_state er_hidden 1.0f grad_hidden_hidden_t |> ignore
sgemm2 nT T (-inv_batch_size*learning_rate) d_training_sequence1.Value er_hidden 1.0f grad_input_hidden_t |> ignore

let grad_bias_out_t = createEmptyAndSetZero grad_bias_out
let grad_bias_hidden_t = createEmptyAndSetZero grad_bias_hidden

calculateBias (-inv_batch_size*learning_rate) er_out momentum_rate grad_bias_out_t
calculateBias (-inv_batch_size*learning_rate) er_hidden2 momentum_rate grad_bias_hidden_t
calculateBias (-inv_batch_size*learning_rate) er_hidden 1.0f grad_bias_hidden_t

//let t1 = grad_bias_hidden_t.dArray.Gather()
//let t2 = grad_bias_hidden.dArray.Gather()
//let t = Array.map2(fun a b -> abs(a-b)) t1 t2
//t |> Array.max

//let inv_batch_size = 1.0f / float32 d_training_sequence1.Value.num_cols
// A simple test whether it works. The same as for the basic_rnn.
let unfolded_rnn num_iterations learning_coef momentum_rate =
    let a1 = gradient_step_forward weight_pars d_training_sequence1 None None
    let a2 = gradient_step_forward weight_pars d_training_sequence2 (Some a1) None
    let b2 = gradient_step_forward weight_pars2 (Some a2) None None

    let empty_state = createEmptyAndSetZero a1

    for i=1 to num_iterations do
        (*
        // Add Nesterov's momentum
        sgeam2 nT nT momentum_rate grad_pars.grad_bias_hidden 1.0f weight_pars.bias_hidden weight_pars.bias_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_pars2.grad_bias_hidden 1.0f weight_pars2.bias_hidden weight_pars2.bias_hidden |> ignore

        sgeam2 nT nT momentum_rate grad_pars.grad_input_hidden 1.0f weight_pars.weights_input_hidden weight_pars.weights_input_hidden |> ignore
        sgeam2 nT nT momentum_rate grad_pars.grad_hidden_hidden.Value 1.0f weight_pars.weights_hidden_hidden.Value weight_pars.weights_hidden_hidden.Value |> ignore
        sgeam2 nT nT momentum_rate grad_pars2.grad_input_hidden 1.0f weight_pars2.weights_input_hidden weight_pars2.weights_input_hidden |> ignore
        *)

        // Forward Steps
        let a1 = gradient_step_forward weight_pars d_training_sequence1 None (Some a1)
        let a2 = gradient_step_forward weight_pars d_training_sequence2 (Some a1) (Some a2)
        let b2 = gradient_step_forward weight_pars2 (Some a2) None (Some b2)

        // Forward pass.
        let xh = sgemm2 T nT 1.0f weights_input_hidden d_training_sequence1.Value 0.0f xh
        let hh = sgemm2 T nT 1.0f weights_hidden_hidden prev_hidden_state 0.0f hh

        let combined_hidden = sgeam2 nT nT 1.0f xh 1.0f hh combined_hidden
        addBias combined_hidden bias_hidden
        reluActivationModule.Apply(combined_hidden,combined_hidden) |> ignore

        let xh2 = sgemm2 T nT 1.0f weights_input_hidden d_training_sequence2.Value 0.0f xh2
        let hh2 = sgemm2 T nT 1.0f weights_hidden_hidden combined_hidden 0.0f hh2

        let combined_hidden2 = sgeam2 nT nT 1.0f xh2 1.0f hh2 combined_hidden2
        addBias combined_hidden2 bias_hidden
        reluActivationModule.Apply(combined_hidden2,combined_hidden2) |> ignore

        let out = sgemm2 T nT 1.0f weights_hidden_output combined_hidden2 0.0f out
        addBias out bias_out
        reluActivationModule.Apply(out,out) |> ignore

        let target_out =
            Some (d_target_sequence, b2)

        let sq_er = squaredCostModule.Apply(d_target_sequence,b2) * inv_batch_size
        printfn "Squared error cost is %f at iteration %i" sq_er i

        let sq_er2 = squaredCostModule.Apply(d_target_sequence,out) * inv_batch_size
        printfn "Squared error cost is %f at iteration %i using the original method" sq_er2 i

        // Backprop
        gradient_step_backward weight_pars weight_pars2 grad_pars grad_pars2 None target_out None a2 d_training_sequence2 learning_coef momentum_rate
        gradient_step_backward weight_pars weight_pars2 grad_pars grad_pars2 None None (Some a2) a1 d_training_sequence1 learning_coef 1.0f
        //gradient_step_backward weight_pars weight_pars2 grad_pars grad_pars2 None None (Some a1) empty_state None learning_coef 1.0f

        // Backwards pass.
        
        let er_out = binaryErrorModule.Apply(d_target_sequence,out,er_out)
        
        let er_hidden2 = sgemm2 nT nT 1.0f weights_hidden_output er_out 0.0f er_hidden2
        binarySparseErrorModule.Apply(er_hidden2,er_hidden2) |> ignore

        let er_hidden = sgemm2 nT nT 1.0f weights_hidden_hidden er_hidden2 0.0f er_hidden
        binarySparseErrorModule.Apply(er_hidden,er_hidden) |> ignore

        // Calculate gradients

        sgemm2 nT T (-inv_batch_size*learning_rate) combined_hidden2 er_out momentum_rate grad_hidden_output_t |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) combined_hidden er_hidden2 momentum_rate grad_hidden_hidden_t |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) d_training_sequence2.Value er_hidden2 momentum_rate grad_input_hidden_t |> ignore

        sgemm2 nT T (-inv_batch_size*learning_rate) prev_hidden_state er_hidden 1.0f grad_hidden_hidden_t |> ignore
        sgemm2 nT T (-inv_batch_size*learning_rate) d_training_sequence1.Value er_hidden 1.0f grad_input_hidden_t |> ignore

        calculateBias (-inv_batch_size*learning_rate) er_out momentum_rate grad_bias_out_t
        calculateBias (-inv_batch_size*learning_rate) er_hidden2 momentum_rate grad_bias_hidden_t
        //calculateBias (-inv_batch_size*learning_rate) er_hidden 1.0f grad_bias_hidden_t

        (*
        // Remove Nesterov's momentum
        sgeam2 nT nT -momentum_rate grad_pars.grad_bias_hidden 1.0f weight_pars.bias_hidden weight_pars.bias_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_pars2.grad_bias_hidden 1.0f weight_pars2.bias_hidden weight_pars2.bias_hidden |> ignore

        sgeam2 nT nT -momentum_rate grad_pars.grad_input_hidden 1.0f weight_pars.weights_input_hidden weight_pars.weights_input_hidden |> ignore
        sgeam2 nT nT -momentum_rate grad_pars.grad_hidden_hidden.Value 1.0f weight_pars.weights_hidden_hidden.Value weight_pars.weights_hidden_hidden.Value |> ignore
        sgeam2 nT nT -momentum_rate grad_pars2.grad_input_hidden 1.0f weight_pars2.weights_input_hidden weight_pars2.weights_input_hidden |> ignore

        // Gradient clipping.
        for x in [|grad_pars2.grad_input_hidden;grad_pars.grad_hidden_hidden.Value;grad_pars.grad_input_hidden;grad_pars.grad_bias_hidden;grad_pars2.grad_bias_hidden|] do
            gradclipModule.Apply(1.0f,x,x) |> ignore
        *)

        // Add gradients
        (*
        sgeam2 nT nT 1.0f weight_pars.weights_input_hidden 1.0f grad_pars.grad_input_hidden weight_pars.weights_input_hidden |> ignore
        sgeam2 nT nT 1.0f weight_pars.weights_hidden_hidden.Value 1.0f grad_pars.grad_hidden_hidden.Value weight_pars.weights_hidden_hidden.Value |> ignore
        sgeam2 nT nT 1.0f weight_pars2.weights_input_hidden 1.0f grad_pars2.grad_input_hidden weight_pars2.weights_input_hidden |> ignore

        sgeam2 nT nT 1.0f weight_pars.bias_hidden 1.0f grad_pars.grad_bias_hidden weight_pars.bias_hidden |> ignore
        sgeam2 nT nT 1.0f weight_pars2.bias_hidden 1.0f grad_pars2.grad_bias_hidden weight_pars2.bias_hidden |> ignore
        *)

        
        let er1 = squaredCostModule.Apply(grad_pars.grad_input_hidden,grad_input_hidden_t)
        let er2 = squaredCostModule.Apply(grad_pars.grad_hidden_hidden.Value,grad_hidden_hidden_t)
        let er3 = squaredCostModule.Apply(grad_pars2.grad_input_hidden,grad_hidden_output_t)
        let er4 = squaredCostModule.Apply(grad_pars.grad_bias_hidden,grad_bias_hidden_t)
        let er5 = squaredCostModule.Apply(grad_pars2.grad_bias_hidden,grad_bias_out_t)

        printfn "er1=%f" er1
        printfn "er2=%f" er2
        printfn "er3=%f" er3
        printfn "er4=%f" er4
        printfn "er5=%f" er5
        
        
        sgeam2 nT nT 1.0f weight_pars.weights_input_hidden 1.0f grad_input_hidden_t weight_pars.weights_input_hidden |> ignore
        sgeam2 nT nT 1.0f weight_pars.weights_hidden_hidden.Value 1.0f grad_hidden_hidden_t weight_pars.weights_hidden_hidden.Value |> ignore
        sgeam2 nT nT 1.0f weight_pars2.weights_input_hidden 1.0f grad_hidden_output_t weight_pars2.weights_input_hidden |> ignore

        sgeam2 nT nT 1.0f weight_pars.bias_hidden 1.0f grad_bias_hidden_t weight_pars.bias_hidden |> ignore
        sgeam2 nT nT 1.0f weight_pars2.bias_hidden 1.0f grad_bias_out_t weight_pars2.bias_hidden |> ignore
        

//let learning_rate = 0.1f
//let learning_coef = (-inv_batch_size*learning_rate)
//let momentum_rate = 0.9f

unfolded_rnn 100 learning_coef momentum_rate
