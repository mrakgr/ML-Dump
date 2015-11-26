// It is retarded how much difficulty this is giving me. I am going to go through every step by hand. 
// It worked in Tensorflow so I am absolutely sure that a simple RNN should be able to do this task.
// Why the hell is is always exploding?

// The sequence recall with a standard RNN.
// The LSTM is killing me so I will try a standard RNN.
// Hopefully nothing is wrong with my standard RNN implementation.

// I have tested RNN and GRU thoroughly in Tensorflow and my current hypothesis is that I
// am getting blowups is because in the final layer some of the outputs get to close too zero or one.

// Edit: I was right on the above, but it turned out there was an error in constrained_clipped_linear_sigmoid 
// that sent it to 0.0f or 1.0f in the extreme.

#load "rnn_lstm_v2.fsx"
open Rnn_lstm_v2
open Rnn_standard_v2
open Utils.Utils

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

//let t1 = training_data_transposed.[0..(batch_size*7)-1]
//let t2 = training_data_transposed.[19*(batch_size*7)..20*(batch_size*7)-1]

//let c = Array.forall2 (fun a b -> a = b) t1 t2

let d_training_data =
    [|
    for i=0 to target_length-1 do
        yield ({num_rows=7;num_cols=batch_size;dArray=worker.Malloc(training_data_transposed.[i*(batch_size*7)..(i+1)*(batch_size*7)-1])}:dM) |]

let hidden_size = 10
let input_size = 7

let xh = createRandomMatrix hidden_size input_size
let hh = createRandomMatrix hidden_size hidden_size
let hy = createRandomMatrix input_size hidden_size

let bias_h = createRandomMatrix hidden_size 1
let bias_y = createRandomMatrix input_size 1

let mom_xh = createEmptyAndSetZero xh
let mom_hh = createEmptyAndSetZero hh
let mom_hy = createEmptyAndSetZero hy

let mom_bias_h = createEmptyAndSetZero bias_h
let mom_bias_y = createEmptyAndSetZero bias_y

let momentum_rate = 0.9f
let learning_rate = 0.1f
let learning_coef = -learning_rate/float32 batch_size

let num_iterations = 600
for i=1 to num_iterations do
    let a1_xh = sgemm nT nT 1.0f xh d_training_data.[0]
    addBias a1_xh bias_h
    let a1 = clippedLinearTanhActivationModule.Apply(a1_xh)

    let a2_hh = sgemm nT nT 1.0f hh a1
    let a2_xh = sgemm nT nT 1.0f xh d_training_data.[1]
    let a2_preact = sgeam nT nT 1.0f a2_hh 1.0f a2_xh
    addBias a2_preact bias_h
    let a2 = clippedLinearTanhActivationModule.Apply(a2_preact)

    let a3_hh = sgemm nT nT 1.0f hh a2
    addBias a3_hh bias_h
    let a3 = clippedLinearTanhActivationModule.Apply(a3_hh)

    let y_hy = sgemm nT nT 1.0f hy a3
    addBias y_hy bias_y
    let y = constrainedClippedLinearLogisticActivationModule.Apply(y_hy)

    printfn "%f" (crossEntropyCostModule.Apply(d_training_data.[0],y)/float32 batch_size)

    let er_y = binaryErrorModule.Apply(d_training_data.[0],y)
    calculateBias learning_coef er_y momentum_rate mom_bias_y
    let gr_y_hy = sgemm nT T learning_coef er_y a3

    let er_a3_preact = sgemm T nT 1.0f hy er_y
    let er_a3 = clippedLinearTanhErrorModule.Apply(er_a3_preact,a3)
    calculateBias learning_coef er_a3 momentum_rate mom_bias_h
    let gr_a3_hh = sgemm nT T learning_coef er_a3 a2

    let er_a2_preact = sgemm T nT 1.0f hh er_a3
    let er_a2 = clippedLinearTanhErrorModule.Apply(er_a2_preact,a2)
    calculateBias learning_coef er_a2 1.0f mom_bias_h
    let gr_a2_hh = sgemm nT T learning_coef er_a2 a1
    let gr_a2_xh = sgemm nT T learning_coef er_a2 d_training_data.[1]

    let er_a1_preact = sgemm T nT 1.0f hh er_a2
    let er_a1 = clippedLinearTanhErrorModule.Apply(er_a1_preact,a1)
    calculateBias learning_coef er_a1 1.0f mom_bias_h
    let gr_a1_xh = sgemm nT T learning_coef er_a1 d_training_data.[0]

    let gr_xh = sgeam nT nT 1.0f gr_a1_xh 1.0f gr_a2_xh
    let gr_hh = sgeam nT nT 1.0f gr_a2_hh 1.0f gr_a3_hh
    let gr_hy = gr_y_hy

    sgeam2 nT nT momentum_rate mom_xh 1.0f gr_xh mom_xh |> ignore
    sgeam2 nT nT momentum_rate mom_hh 1.0f gr_hh mom_hh |> ignore
    sgeam2 nT nT momentum_rate mom_hy 1.0f gr_hy mom_hy |> ignore

    sgeam2 nT nT 1.0f mom_xh 1.0f xh xh |> ignore
    sgeam2 nT nT 1.0f mom_hh 1.0f hh hh |> ignore
    sgeam2 nT nT 1.0f mom_hy 1.0f hy hy |> ignore

    sgeam2 nT nT 1.0f mom_bias_h 1.0f bias_h bias_h |> ignore
    sgeam2 nT nT 1.0f mom_bias_y 1.0f bias_y bias_y |> ignore