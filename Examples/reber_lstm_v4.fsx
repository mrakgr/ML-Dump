// Hopefully with the redesigned library, things will go more smoothly this time.

// Nope.
// While the gradients do not blow up, the errors do. What the hell do I do about this?

#load "rnn_lstm_v2.fsx"
open Rnn_lstm_v2
open Rnn_standard_v2
open Utils.Utils

#load "embedded_reber.fsx"
open Embedded_reber

let training_data = make_reber_set 3000

let target_length = 20

let twenties = training_data |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
let batch_size = (twenties |> Seq.length)

let d_training_data =
    [|
    for i=0 to target_length-2 do
        let input = [|
            for k=0 to batch_size-1 do
                let example = twenties.[k]
                let s, input, output = example
                yield input.[i]|] |> Array.concat

        let output = [|
            for k=0 to batch_size-1 do
                let example = twenties.[k]
                let s, input, output = example
                yield output.[i+1]|] |> Array.concat

        let t1 = {num_rows=7;num_cols=batch_size;dArray=worker.Malloc(input)}:dM
        let t2 = {num_rows=7;num_cols=batch_size;dArray=worker.Malloc(output)}:dM
        yield t1,t2 |]

let hidden_size = 100
let input_size = 7

let l1 = createRandomLstmCell hidden_size input_size
let g1 = createGradsLikeLSTM l1

let l2 = createRandomFeedforwardWeights hidden_size input_size
let g2 = createGradsLike l2

let longest_string = target_length

let input0, output0 = d_training_data.[0]

let learning_rate = 0.1f
let learning_coef = -learning_rate / float32 batch_size
let momentum_rate = 0.9f

// Preallocating memory.
// Going from 2..longest_string gives me longest_string-1 allocations.
// I purposely ommit the last step because it has no predictions.
let activations_lstm = [|for i=0 to longest_string-2 do yield lstm_activation l1 (Some input0) None None|]
let outputs_standard = [|for i=0 to longest_string-2 do yield rnn_forward l2 (Some activations_lstm.[0].block_output) None None logisticActivationModule|]
let errors_standard = [|for i=0 to longest_string-2 do yield createEmptyMatrixLike outputs_standard.[i]|]
let errors_lstm = [|for i=0 to longest_string-2 do yield createErrorsLikeActivationsLSTM activations_lstm.[i]|]

#time
let num_iterations = 100
for i=1 to num_iterations do
    let mutable error = 0.0f

    let input0, output0 = d_training_data.[0]
        
    // The first step has to be done separately as it does not take inputs from previous timesteps.
    lstm_activation l1 (Some input0) None (Some activations_lstm.[0]) |> ignore
    rnn_forward l2 (Some activations_lstm.[0].block_output) None (Some outputs_standard.[0]) logisticActivationModule |> ignore

    //error <- error + crossEntropyCostModule.Apply(output0,outputs_standard.[0])
    error <- error + squaredCostModule.Apply(output0,outputs_standard.[0])
    
    for i=1 to target_length-2 do
        let input_i, output_i = d_training_data.[i]
        // The following steps are made inside the loop.
        lstm_activation l1 (Some input_i) (Some activations_lstm.[i-1]) (Some activations_lstm.[i]) |> ignore
        rnn_forward l2 (Some activations_lstm.[i].block_output) None (Some outputs_standard.[i]) logisticActivationModule |> ignore

        //error <- error + crossEntropyCostModule.Apply(output_i,outputs_standard.[i])
        error <- error + squaredCostModule.Apply(output_i,outputs_standard.[i])
    let input_last, output_last = d_training_data.[target_length-2]
    
    rnn_backward_error_top output_last outputs_standard.[target_length-2] errors_standard.[target_length-2] |> ignore
        
    // The gradient for the feedforward last layer.
    rnn_backwards_weight (Some (errors_standard.[target_length-2], l2, g2,true)) None activations_lstm.[target_length-2].block_output learning_coef momentum_rate

    rnn_backward_error_middle (Some (errors_standard.[target_length-2], l2)) None (activations_lstm.[target_length-2].block_output, errors_lstm.[target_length-2].error_block_output) logisticErrorModule |> ignore
    lstm_error_feedforward_layer None (Some activations_lstm.[target_length-3]) activations_lstm.[target_length-2] l1 errors_lstm.[target_length-2] |> ignore

    weight_input_grads learning_coef errors_lstm.[target_length-2] input_last momentum_rate g1
    weight_biases_grad learning_coef errors_lstm.[target_length-2] momentum_rate g1
    weight_peephole_grads learning_coef activations_lstm.[target_length-2] None errors_lstm.[target_length-2] momentum_rate g1
        
    let mutable momentum_rate = momentum_rate
    for i=target_length-3 downto 0 do
        let input_i, output_i = d_training_data.[i]

        rnn_backward_error_top output_i outputs_standard.[i] errors_standard.[i] |> ignore

        // The gradient for the feedforward last layer. I forgot to use this line in the previous file. Damn
        rnn_backwards_weight (Some (errors_standard.[i], l2, g2, true)) None activations_lstm.[i].block_output learning_coef momentum_rate

        rnn_backward_error_middle (Some (errors_standard.[i], l2)) None (activations_lstm.[i].block_output, errors_lstm.[i].error_block_output) logisticErrorModule |> ignore
        lstm_error_feedforward_layer (Some (activations_lstm.[i+1],errors_lstm.[i+1])) (if i > 0 then Some activations_lstm.[i-1] else None) activations_lstm.[i] l1 errors_lstm.[i] |> ignore

        weight_input_grads learning_coef errors_lstm.[i] input_i 1.0f g1
        weight_biases_grad learning_coef errors_lstm.[i] 1.0f g1
        weight_peephole_grads learning_coef activations_lstm.[i] (Some errors_lstm.[i+1]) errors_lstm.[i] 1.0f g1

        weight_hidden_grads learning_coef errors_lstm.[i+1] activations_lstm.[i].block_output momentum_rate g1

        momentum_rate <- 1.0f

    applyGradientClippingLSTM g1 1.0f
    applyGradientClipping g2 1.0f
    (*
    sumGradsLSTM g1
    sumWeightsLSTM l1

    sumGrads g2
    sumWeights l2

    sumErrors errors_lstm.[18]
    sumErrors errors_lstm.[17]
    *)
    // Add gradients
    addGradsToWeights 1.0f g2 l2 l2
    addGradsToWeightsLSTM l1 g1

    printfn "The cross entropy errors after epoch %i is %f" i (error/float32 batch_size)
#time

