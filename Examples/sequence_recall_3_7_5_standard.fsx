// The sequence recall with a standard RNN.
// The LSTM is killing me so I will try a standard RNN.
// Hopefully nothing is wrong with my standard RNN implementation.

// I have tested RNN and GRU thoroughly in Tensorflow and my current hypothesis is that I
// am getting blowups is because in the final layer some of the outputs get to close too zero or one.

#load "rnn_lstm_v2.fsx"
open Rnn_lstm_v2
open Rnn_standard_v2
open Utils.Utils

let rng = System.Random()
let sequence_recall_data batch_size seq_length =
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
let batch_size = 5
let training_data = sequence_recall_data batch_size target_length
let training_data_transposed =
    [|
    for i=0 to target_length-1 do
        for k=0 to batch_size-1 do
            let ind = k*target_length+i
            yield training_data.[ind] |] |> Array.concat

//let t1 = training_data_transposed.[0..(batch_size*7)-1]
//let t2 = training_data_transposed.[19*(batch_size*7)..20*(batch_size*7)-1]

//let c = Array.forall2 (fun a b -> a = b) t1 t2

let d_training_data =
    [|
    for i=0 to target_length-1 do
        yield ({num_rows=7;num_cols=batch_size;dArray=worker.Malloc(training_data_transposed.[i*(batch_size*7)..(i+1)*(batch_size*7)-1])}:dM) |]

let hidden_size = 10
let input_size = 7

let l1 = createRandomRecurrentWeights input_size hidden_size
let g1 = createGradsLike l1

let acts = [|for i=0 to target_length-1 do yield rnn_forward l1 (Some d_training_data.[0]) None None clippedLinearTanhActivationModule|]
let ers = [|for i=0 to target_length-1 do yield createEmptyMatrixLike acts.[i] |]

let l2 = createRandomFeedforwardWeights hidden_size input_size
let g2 = createGradsLike l2

let actsF = rnn_forward l2 (Some acts.[0]) None None constrainedClippedLinearLogisticActivationModule
let ersF = createEmptyMatrixLike actsF

let learning_rate = 0.1f
let learning_coef = -learning_rate / float32 batch_size
let momentum_rate = 0.9f

let num_iterations = 500
for i=1 to num_iterations do
    for k=0 to target_length-1 do
        rnn_forward l1 (if k < target_length-1 then Some d_training_data.[k] else None) (if k > 0 then Some acts.[k-1] else None) (Some acts.[k]) clippedLinearTanhActivationModule |> ignore

    rnn_forward l2 (Some acts.[0]) None (Some actsF) constrainedClippedLinearLogisticActivationModule |> ignore

    rnn_backward_error_top d_training_data.[target_length-1] actsF ersF |> ignore
    
    rnn_backwards_weight (Some (ersF, l2, g2,true)) None acts.[target_length-1] learning_coef momentum_rate

    printfn "Squared error cost at epoch %i is %f" i (crossEntropyCostModule.Apply(d_training_data.[target_length-1],actsF) / float32 batch_size)
    
    rnn_backward_error_middle (Some (ersF, l2)) None (acts.[target_length-1], ers.[target_length-1]) clippedLinearTanhErrorModule |> ignore
    
    let mutable momentum_rate = momentum_rate
    for k=target_length-2 downto 0 do
        rnn_backward_error_middle None (Some (acts.[k+1],l1)) (acts.[k], ers.[k]) clippedLinearTanhErrorModule |> ignore
        
        rnn_backwards_weight None (Some (ers.[k], l1, g1,true)) acts.[k] learning_coef momentum_rate
        rnn_backwards_weight (Some (ers.[k], l1, g1,false)) None d_training_data.[k] learning_coef momentum_rate

        momentum_rate <- 1.0f
    
    addGradsToWeights 1.0f g2 l2 l2
    addGradsToWeights 1.0f g1 l1 l1
