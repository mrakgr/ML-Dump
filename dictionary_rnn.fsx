// I've decided to redesign the standard RNN using Dictionary-ies.
// My previous design is so bad and it is not even correct.

// Hopefully I will be able to eliminate a host of bugs using this new
// approach.

#load "rnn_standard_v3.fsx"
open Rnn_standard_v3
open Utils.Utils
open System.Collections.Generic

let rng = System.Random()

// To me these two problems look roughly similar but to the network they are worlds apart it seems.
let sequence_recall_data batch_size seq_length =
    [|
    for k = 1 to batch_size do
        let t = [|for i=1 to 7 do yield if rng.NextDouble() > 0.5 then 1.0f else 0.0f|]
        yield t
        for i=2 to seq_length-1 do
            let t = [|for i=1 to 7 do yield if rng.NextDouble() > 0.5 then 1.0f else 0.0f|]
            yield t
        yield t |]

let sequence_recall_data2 batch_size seq_length =
    [|
    for k = 1 to batch_size do
        let e = rng.NextDouble()*7.0 |> int
        let t = [|0.0f;0.0f;0.0f;0.0f;0.0f;0.0f;0.0f;|]
        t.[e] <- 0.5f
        yield t
        for i=2 to seq_length-1 do
            let e = rng.NextDouble()*7.0 |> int
            let t = [|0.0f;0.0f;0.0f;0.0f;0.0f;0.0f;0.0f;|]
            t.[e] <- 0.5f
            yield t
        yield t |]

let target_length = 3
let batch_size = 50
let training_data = sequence_recall_data batch_size target_length
let training_data_transposed =
    [|
    for i=0 to target_length-1 do
        for k=0 to batch_size-1 do
            let ind = k*target_length+i
            yield training_data.[ind] |] |> Array.concat

let d_training_data =
    [|
    for i=0 to target_length-1 do
        yield ({num_rows=7;num_cols=batch_size;dArray=worker.Malloc(training_data_transposed.[i*(batch_size*7)..(i+1)*(batch_size*7)-1])}:dM) |]


let hidden_size = 10
let input_size = 7

let xh = createRandomMatrix hidden_size input_size
let hh = createRandomMatrix hidden_size hidden_size
let bias_h = createRandomMatrix hidden_size 1

let pars1 : weightPars = createWeightPars xh (Some hh) bias_h

let hy = createRandomMatrix input_size hidden_size
let bias_y = createRandomMatrix input_size 1

let pars2 = createWeightPars hy None bias_y

let mom_xh = createEmptyAndSetZero xh
let mom_hh = createEmptyAndSetZero hh
let mom_hy = createEmptyAndSetZero hy

let mom_bias_h = createEmptyAndSetZero bias_h
let mom_bias_y = createEmptyAndSetZero bias_y

let momentum_rate = 0.9f
let learning_rate = 0.3f
let learning_coef = -learning_rate/float32 batch_size

let forward_dict = new Dictionary<int*int,dM>()
let error_dict = new Dictionary<int*int,dM>()
let label_dict = new Dictionary<int*int,dM>()
let pars_dict = new Dictionary<int,weightPars>(5)

pars_dict.Add(1,pars1)
pars_dict.Add(2,pars2)

forward_dict.Add((0,1),d_training_data.[0])
forward_dict.Add((0,2),d_training_data.[1])
label_dict.Add((2,3),d_training_data.[2])

let optional_get (dict: Dictionary<'a,'b>) key =
    if dict.ContainsKey(key) then Some dict.[key] else None

let rnn_forward row col (activation_module: DeviceUnaryTransformModule<float32>) =
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

let rnn_error_label() =
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

let rnn_error row col (errorModule: DeviceBinaryTransformModule<float32>) =
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

let rnn_set_momentum_flags() =
    for x in pars_dict do
        x.Value.momentum_flag_input := momentum_rate
        x.Value.momentum_flag_hidden := momentum_rate
        x.Value.momentum_flag_bias := momentum_rate

let rnn_gradient_calculate row col =
    let act_left = optional_get forward_dict (row,col-1)    
    let act_down = optional_get forward_dict (row-1,col)

    let er_cur = error_dict.[row,col]
    let pars_cur = pars_dict.[row]
        
    let t = dynamic_multiply nT T learning_coef (Some er_cur) act_left pars_cur.momentum_flag_hidden pars_cur.momentum_weights_hidden_hidden
    let t = dynamic_multiply nT T learning_coef (Some er_cur) act_down pars_cur.momentum_flag_input (Some pars_cur.momentum_weights_input_hidden)
    dynamicCalculateBias learning_coef er_cur pars_cur.momentum_flag_bias pars_cur.momentum_bias_hidden
    ()

let rnn_gradient_add_to_weights() =
    for x in pars_dict do
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_weights_input_hidden) 1.0f (Some x.Value.weights_input_hidden) (Some x.Value.weights_input_hidden)
        let t = dynamic_add nT nT 1.0f x.Value.momentum_weights_hidden_hidden 1.0f x.Value.weights_hidden_hidden x.Value.weights_hidden_hidden
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_bias_hidden) 1.0f (Some x.Value.bias_hidden) (Some x.Value.bias_hidden)
        ()

/// Adds the momentum to the copy matrices. Used in Nesterov's Momentum.
let rnn_gradient_add_to_weights_nestorov() =
    for x in pars_dict do
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_weights_input_hidden) 1.0f (Some x.Value.weights_input_hidden_copy) (Some x.Value.weights_input_hidden_copy)
        let t = dynamic_add nT nT 1.0f x.Value.momentum_weights_hidden_hidden 1.0f x.Value.weights_hidden_hidden_copy x.Value.weights_hidden_hidden_copy
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_bias_hidden) 1.0f (Some x.Value.bias_hidden_copy) (Some x.Value.bias_hidden_copy)
        ()

let rnn_overwrite_with_copies_and_add_momentum() =
    for x in pars_dict do
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_weights_input_hidden) 1.0f (Some x.Value.weights_input_hidden_copy) (Some x.Value.weights_input_hidden)
        let t = dynamic_add nT nT 1.0f x.Value.momentum_weights_hidden_hidden 1.0f x.Value.weights_hidden_hidden_copy x.Value.weights_hidden_hidden
        let t = dynamic_add nT nT 1.0f (Some x.Value.momentum_bias_hidden) 1.0f (Some x.Value.bias_hidden_copy) (Some x.Value.bias_hidden)
        ()

let num_iterations = 600
for i=1 to num_iterations do
    
    // Adds copy+momentum matrices are assigned to the weights. Nesterov's Momentum.
    rnn_overwrite_with_copies_and_add_momentum()

    rnn_forward 1 1 clippedLinearTanhActivationModule
    rnn_forward 1 2 clippedLinearTanhActivationModule
    rnn_forward 1 3 clippedLinearTanhActivationModule

    rnn_forward 2 3 constrainedClippedLinearLogisticActivationModule

    printfn "%f" (crossEntropyCostModule.Apply(d_training_data.[0],forward_dict.[2,3])/float32 batch_size)

    rnn_error_label()
    
    rnn_error 1 3 clippedLinearTanhErrorModule
    rnn_error 1 2 clippedLinearTanhErrorModule
    rnn_error 1 1 clippedLinearTanhErrorModule

    rnn_set_momentum_flags()

    rnn_gradient_calculate 2 3
    rnn_gradient_calculate 1 3
    rnn_gradient_calculate 1 2
    rnn_gradient_calculate 1 1
   
    rnn_gradient_add_to_weights_nestorov()

