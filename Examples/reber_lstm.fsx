// Library file for the LSTM functions.
// As the LSTM is quite large, it makes sense to pull out the components so I can call them from elsewhere.

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
        a*(log b_min) + (1.0f-a)*log (1.0f - b_max)@>

let hidden_size = 10
let input_size = 7
let batch_size = 1

let l1 = createRandomLstmCell hidden_size batch_size input_size
let g1 = createGradsLikeLSTM l1

let l2 = createRandomFeedforwardWeights hidden_size input_size
let g2 = createGradsLike l2

#load "embedded_reber.fsx"
open Embedded_reber

let training_data = make_reber_set 1000

let d_training_data = [|
    for s,input, output in training_data do
        let inp = 
            [| for x in input do
                    let t = {num_rows=7;num_cols=1;dArray=worker.Malloc(x)} : dM
                    yield t|]
        let pred = 
            [| for x in output.[1..] do
                    let t = {num_rows=7;num_cols=1;dArray=worker.Malloc(x)} : dM
                    yield t|]
        yield inp,pred|]

let longest_string =
    let mutable l = 0
    for s,input, output in training_data do
        if s.Length > l then l <- s.Length
    l

let input, output = d_training_data.[0]
let input0 = input.[0]
let output0 = output.[0]

let learning_rate = 0.1f
let learning_coef = -learning_rate / float32 batch_size
let momentum_rate = 0.9f

// Preallocating memory.
// Going from 2..longest_string gives me longest_string-1 allocations.
// I purposely ommit the last step because it has no predictions.
let activations_lstm = [|for i=2 to longest_string do yield lstm_activation_allocation l1 (Some input0) None|]
let outputs_standard = [|for i=2 to longest_string do yield rnn_forward l2 (Some activations_lstm.[0].block_output) None None logisticActivationModule|]
let errors_standard = [|for i=2 to longest_string do yield rnn_backward_error_top output0 outputs_standard.[0] None|]

let errors_lstm_up = [|for i=2 to longest_string do yield rnn_backward_error_middle (Some (errors_standard.[0], l2)) None None activations_lstm.[0].block_output logisticErrorModule |]
let errors_lstm = [|for i=2 to longest_string do yield lstm_error_allocation_cell (Some errors_lstm_up.[0]) None activations_lstm.[0] l1|]


#time
let num_iterations = 10
for i=1 to num_iterations do
    let mutable c = 0
    let mutable error = 0.0f
    for input,output in d_training_data do
        if c % 100 = 0 then printfn "%i/1000..." c
        c <- c+1
        // The first step has to be done separately as it does not take inputs from previous timesteps.
        lstm_activation l1 (Some input.[0]) None activations_lstm.[0] |> ignore
        rnn_forward l2 (Some activations_lstm.[0].block_output) None (Some outputs_standard.[0]) logisticActivationModule |> ignore

        error <- error - crossEntropyCostModule.Apply(output.[0],outputs_standard.[0])
        
        for i=1 to output.Length-1 do
            // The following steps are made inside the loop.
            lstm_activation l1 (Some input.[i]) (Some activations_lstm.[i-1].block_output) activations_lstm.[i] |> ignore
            rnn_forward l2 (Some activations_lstm.[i].block_output) None (Some outputs_standard.[i]) logisticActivationModule |> ignore

            error <- error - crossEntropyCostModule.Apply(output.[i],outputs_standard.[i])

        rnn_backward_error_top output.[output.Length-1] outputs_standard.[output.Length-1] (Some errors_standard.[output.Length-1]) |> ignore
        
        // The gradient for the feedforward last layer.
        rnn_backwards_weight (Some (errors_standard.[output.Length-1], l2, g2,true)) None activations_lstm.[output.Length-1].block_output learning_coef momentum_rate

        rnn_backward_error_middle (Some (errors_standard.[output.Length-1], l2)) None (Some errors_lstm_up.[output.Length-1]) activations_lstm.[output.Length-1].block_output logisticErrorModule |> ignore
        lstm_error_cell (Some errors_lstm_up.[output.Length-1]) None activations_lstm.[output.Length-1] l1 errors_lstm.[output.Length-1] |> ignore

        weight_input_grads learning_coef errors_lstm.[output.Length-1] input.[output.Length-1] momentum_rate g1
        weight_biases_grad learning_coef errors_lstm.[output.Length-1] momentum_rate g1
        weight_peephole_grads learning_coef activations_lstm.[output.Length-1] None errors_lstm.[output.Length-1] 1.0f g1
        
        let mutable momentum_rate = momentum_rate
        for i=output.Length-2 downto 0 do
            rnn_backward_error_top output.[i] outputs_standard.[i] (Some errors_standard.[i]) |> ignore
            rnn_backward_error_middle (Some (errors_standard.[i], l2)) None (Some errors_lstm_up.[i]) activations_lstm.[i].block_output logisticErrorModule |> ignore
            lstm_error_cell (Some errors_lstm_up.[i]) (Some (activations_lstm.[i+1],l1,errors_lstm.[i+1])) activations_lstm.[i] l1 errors_lstm.[i] |> ignore

            weight_input_grads learning_coef errors_lstm.[i] input.[i] 1.0f g1
            weight_biases_grad learning_coef errors_lstm.[i] 1.0f g1
            weight_peephole_grads learning_coef activations_lstm.[i] (Some errors_lstm.[i+1]) errors_lstm.[i] 1.0f g1

            weight_hidden_grads learning_coef errors_lstm.[i+1] activations_lstm.[i].block_output momentum_rate g1

            momentum_rate <- 1.0f

        // Add gradients
        addGradsToWeights 1.0f g2 l2 l2
        addGradsToWeightsLSTM l1 g1

    printfn "The cross entropy errors after epoch %i is %f" i (error/1000.0f)
        
#time