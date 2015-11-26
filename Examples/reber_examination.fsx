// I have no idea why the LSTM keep blowing up on Reber.
// I'll use this file just to examine the data I am generating.

// This sucks balls. It took me an entire week to get that brilliant
// flash of insight and fix the LSTM, but it still was not enough.

// I have no idea where I am going wrong.

// ...I discovered just now that I have the sequences reversed.

#load "rnn_lstm.fsx"
open Rnn_lstm
open Rnn_standard
open Utils.Utils

//#load "rnn_standard.fsx"
//open Rnn_standard

#load "embedded_reber.fsx"
open Embedded_reber

let training_data = make_reber_set 3000

let target_length = 20

let twenties = training_data |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
let batch_size = (twenties |> Seq.length)

let d_training_data =
    [|
    for i=19 downto 0 do
        let input = [|
            for k=0 to batch_size-1 do 
                let example = twenties.[k]
                let s, input, output = example
                yield input.[i] |] |> Array.concat
        yield input |]

let t = d_training_data.[0]

let d_training_data_pred =
    [|
    for i=0 to 0 do
            for k=0 to 18 do
                let example = twenties.[i]
                let s, input, output = example
                yield output.[k] |]
