// It works, but I do not feel like finishing this. It could be potentially made a lot faster by not having to calculate worthless gradients for the inputs
// and the outputs, but I am not longer interested in this version as the version that can reuse the tape is much faster.

#load "ad_utils_v2.fsx"
open Ad_utils_v2
open FSharp.Charting

#load "embedded_reber.fsx"
open Embedded_reber

let reber_set = make_reber_set 3000

let make_data_from_set target_length =
    let twenties = reber_set |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
    let batch_size = (twenties |> Seq.length)

    let d_training_data =
        [|
        for i=0 to target_length-1 do
            let input = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield input.[i] |] |> Array.concat
            yield RDM.makeNode(7,batch_size,input)|]

    let d_target_data =
        [|
        for i=1 to target_length-1 do // The targets are one less than the inputs. This has the effect of shifting them to the left.
            let output = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield output.[i] |] |> Array.concat
            yield RDM.makeNode(7,batch_size,output)|]

    d_training_data, d_target_data

let train_lstm_reber num_iters learning_rate (data: RDM[]) (targets: RDM[]) clip_coef (l1: LSTMLayer) (l2: Layer) (tape_t: tapeType) (tape_v: tapeType) =
    [|
    let base_nodes = [|l1.ToArray;l2.ToArray|] |> Array.concat
    for x in base_nodes do setModule.A(0.0f,x.r.A,x.r.A)
    let mutable i=1
    let mutable rr=0.0f
    while i <= num_iters && System.Single.IsNaN rr = false do
        tape_t.Clear()
        tape_v.Clear()
        System.GC.Collect()
        let training_loop (data: RDM[]) (targets: RDM[]) (tape_local: tapeType) =
            tape <- tape_local
            let costs = [|
                let mutable a, c = l1.runLayerNoH data.[0]
                let b = l2.runLayerNoH a
                let r = squared_error_cost targets.[0] b
                yield r
    
                for i=1 to data.Length-2 do
                    let a',c' = l1.runLayer data.[i] a c
                    a <- a'; c <- c'
                    let b = l2.runLayerNoH a
                    let r = squared_error_cost targets.[i] b
                    yield r
                |]
            scale (1.0f/float32 costs.Length) (sum_scalars costs)

        let r = training_loop data targets tape_t
        printfn "The cost is %f at iteration %i" r.r.P i

        rr <- r.r.P

        r.r.A <- 1.0f
        // Resets the adjoints to zero.
        for x in base_nodes do setModule.A(0.0f,x.r.A,x.r.A)
        propagateReverseTape()
        tape.Clear()

        // Add gradients.
        for x in base_nodes do
            gradclipModule.A(clip_coef,x.r.A,x.r.A)
            sgeam2 nT nT 1.0f x.r.P -learning_rate x.r.A x.r.P

        i <- i+1
        yield r.r.P |]

let hidden_size = 64

let l1 = LSTMLayer.createRandomLSTMLayer hidden_size 7 tanh_ tanh_
let l2 = Layer.createRandomLayer 7 hidden_size sigmoid

let d_training_data_20, d_target_data_20 = make_data_from_set 20
let d_training_data_19, d_target_data_19 = make_data_from_set 19
let d_training_data_18, d_target_data_18 = make_data_from_set 18
let d_training_data_validation, d_target_data_validation = make_data_from_set 30

let tape_t = tapeType()
let tape_v = tapeType()

let s = [|
        yield train_lstm_reber 33 5.0f d_training_data_20 d_target_data_20 1.0f l1 l2 tape_t tape_v
        |] |> Array.concat

(Chart.Line s).ShowChart()