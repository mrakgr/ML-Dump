// Fuck LSTMs. Let me just try feeding it the shifted input and after that I am done.

//#load "utils.fsx"
#load "rnn_lstm.fsx"
open Rnn_lstm
open Rnn_standard
open Utils.Utils

//#load "rnn_standard.fsx"
//open Rnn_standard

let crossEntropyCostModule = 
    new DeviceBinaryMapReduceModule
        <@ fun a b -> 
        let b_max = min 0.999999f b
        let b_min = max 0.000001f b
        -(a*(log b_min) + (1.0f-a)*log (1.0f - b_max)) @>

#load "embedded_reber.fsx"
open Embedded_reber

let training_data = make_reber_set 3000

let target_length = 20

let twenties = training_data |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
let batch_size = (twenties |> Seq.length)

let d_training_data =
    [|
    for i=target_length-1 downto 0 do
        let input = [|
            for k=0 to batch_size-1 do
                let example = twenties.[k]
                let s, input, output = example
                yield input.[i] |] |> Array.concat

        let t1 = {num_rows=7;num_cols=batch_size;dArray=worker.Malloc(input)}:dM
        yield t1 |]

let hidden_size = 10
let input_size = 7

let l1 = createRandomLstmCell hidden_size input_size
let g1 = createGradsLikeLSTM l1

let l2 = createRandomFeedforwardWeights hidden_size input_size
let g2 = createGradsLike l2

let longest_string = target_length

let input0, output0 = d_training_data.[0], d_training_data.[1]

let learning_rate = 0.1f
let learning_coef = -learning_rate / float32 batch_size
let momentum_rate = 0.9f

// Preallocating memory.
// Going from 2..longest_string gives me longest_string-1 allocations.
// I purposely ommit the last step because it has no predictions.
let activations_lstm = [|for i=1 to longest_string do yield lstm_activation l1 (Some input0) None None|]
let outputs_standard = [|for i=1 to longest_string do yield rnn_forward l2 (Some activations_lstm.[0].block_output) None None logisticActivationModule|]
let errors_standard = [|for i=1 to longest_string do yield rnn_backward_error_top output0 outputs_standard.[0] None|]

let errors_lstm_up = [|for i=1 to longest_string do yield rnn_backward_error_middle (Some (errors_standard.[0], l2)) None None activations_lstm.[0].block_output logisticErrorModule |]
let errors_lstm = [|for i=1 to longest_string do yield lstm_error_cell (Some errors_lstm_up.[0]) None None activations_lstm.[0] l1 None|]

#time
let num_iterations = 500
for i=1 to num_iterations do
    let mutable error = 0.0f

    let input0, output0 = d_training_data.[0], d_training_data.[1]
        
    // The first step has to be done separately as it does not take inputs from previous timesteps.
    lstm_activation l1 (Some input0) None (Some activations_lstm.[0]) |> ignore
    rnn_forward l2 (Some activations_lstm.[0].block_output) None (Some outputs_standard.[0]) logisticActivationModule |> ignore

    //error <- error - crossEntropyCostModule.Apply(output0,outputs_standard.[0])
    error <- error + squaredCostModule.Apply(output0,outputs_standard.[0])
    
    for i=1 to target_length-2 do
        let input_i, output_i = d_training_data.[i], d_training_data.[i+1]
        // The following steps are made inside the loop.
        lstm_activation l1 (Some input_i) (Some activations_lstm.[i-1]) (Some activations_lstm.[i]) |> ignore
        rnn_forward l2 (Some activations_lstm.[i].block_output) None (Some outputs_standard.[i]) logisticActivationModule |> ignore

        //error <- error - crossEntropyCostModule.Apply(output_i,outputs_standard.[i])
        error <- error + squaredCostModule.Apply(output_i,outputs_standard.[i])
    let input_last, output_last = d_training_data.[target_length-2], d_training_data.[target_length-1]
    
    rnn_backward_error_top output_last outputs_standard.[target_length-2] (Some errors_standard.[target_length-2]) |> ignore
        
    // The gradient for the feedforward last layer.
    rnn_backwards_weight (Some (errors_standard.[target_length-2], l2, g2,true)) None activations_lstm.[target_length-2].block_output learning_coef momentum_rate

    rnn_backward_error_middle (Some (errors_standard.[target_length-2], l2)) None (Some errors_lstm_up.[target_length-2]) activations_lstm.[target_length-2].block_output logisticErrorModule |> ignore
    lstm_error_cell (Some errors_lstm_up.[target_length-2]) None (Some activations_lstm.[target_length-3]) activations_lstm.[target_length-2] l1 (Some errors_lstm.[target_length-2]) |> ignore

    weight_input_grads learning_coef errors_lstm.[target_length-2] input_last momentum_rate g1
    weight_biases_grad learning_coef errors_lstm.[target_length-2] momentum_rate g1
    weight_peephole_grads learning_coef activations_lstm.[target_length-2] None errors_lstm.[target_length-2] 1.0f g1
        
    let mutable momentum_rate = momentum_rate
    for i=target_length-3 downto 0 do
        let input_i, output_i = d_training_data.[i],d_training_data.[i+1]

        rnn_backward_error_top output_i outputs_standard.[i] (Some errors_standard.[i]) |> ignore

        // The gradient for the feedforward last layer. I forgot to use this line in the previous file. Damn
        rnn_backwards_weight (Some (errors_standard.[i], l2, g2, true)) None activations_lstm.[i].block_output learning_coef momentum_rate

        rnn_backward_error_middle (Some (errors_standard.[i], l2)) None (Some errors_lstm_up.[i]) activations_lstm.[i].block_output logisticErrorModule |> ignore
        lstm_error_cell (Some errors_lstm_up.[i]) (Some (activations_lstm.[i+1],errors_lstm.[i+1])) (if i > 0 then Some activations_lstm.[i-1] else None) activations_lstm.[i] l1 (Some errors_lstm.[i]) |> ignore

        weight_input_grads learning_coef errors_lstm.[i] input_i 1.0f g1
        weight_biases_grad learning_coef errors_lstm.[i] 1.0f g1
        weight_peephole_grads learning_coef activations_lstm.[i] (Some errors_lstm.[i+1]) errors_lstm.[i] 1.0f g1

        weight_hidden_grads learning_coef errors_lstm.[i+1] activations_lstm.[i].block_output momentum_rate g1

        momentum_rate <- 1.0f

    //applyGradientClippingLSTM g1 0.01f
    //applyGradientClipping g2 0.01f

    // Add gradients
    addGradsToWeights 1.0f g2 l2 l2
    addGradsToWeightsLSTM l1 g1

    printfn "The cross entropy errors after epoch %i is %f" i (error/float32 batch_size)
#time

