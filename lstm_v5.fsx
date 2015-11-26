// Just a test of whether the function are being called correctly.

// Yeah, they are.

//#load "utils.fsx"
#load "rnn_lstm.fsx"
open Rnn_lstm
open Rnn_standard
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
