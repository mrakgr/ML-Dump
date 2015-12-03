#r @"C:\DiffSharp-master\src\DiffSharp\bin\Debug\DiffSharp.dll"
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\FSharp.Quotations.Evaluator.1.0.6\lib\net40\FSharp.Quotations.Evaluator.dll"
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\FSharp.Charting.0.90.13\lib\net40\FSharp.Charting.dll"
#r @"C:\Program Files (x86)\Reference Assemblies\Microsoft\Framework\.NETFramework\v4.6\System.Windows.Forms.DataVisualization.dll"

open DiffSharp.AD.Float32
open DiffSharp.Util

open FSharp.Charting

open System.IO

let rnd = System.Random()

// A layer of neurons
type Layer' =
    {mutable W:DM  // Weight matrix
     mutable b:DV  // Bias vector
     a:DM->DM}     // Activation function

// A feedforward network of neuron layers
type Network' =
    {layers:Layer'[]} // The layers forming this network

let runLayer' (x:DM) (l:Layer') =
    l.W * x + (DM.createCols x.Cols l.b) |> l.a

let runNetwork' (x:DM) (n:Network') =
    Array.fold runLayer' x n.layers

// Backpropagation with SGD and minibatches
// n: network
// eta: learning rate
// epochs: number of training epochs
// mbsize: minibatch size
// loss: loss function
// x: training input matrix
// y: training target matrix
let backprop' (n:Network') (eta:float32) epochs mbsize loss (x:DM) (y:DM) =
    [|
    let i = DiffSharp.Util.GlobalTagger.Next
    let mutable b = 0
    let batches = x.Cols / mbsize
    let mutable j = 0
    while j < epochs do
        b <- 0
        while b < batches do
            let mbX = x.[*, (b * mbsize)..((b + 1) * mbsize - 1)]
            let mbY = y.[*, (b * mbsize)..((b + 1) * mbsize - 1)]

            for l in n.layers do
                l.W <- l.W |> makeReverse i
                l.b <- l.b |> makeReverse i

            let L:D = loss (runNetwork' mbX n) mbY
            L |> reverseProp (D 1.0f)

            printfn "%A" L

            for l in n.layers do
                l.W <- (l.W.P - eta * l.W.A)
                l.b <- (l.b.P - eta * l.b.A)

            printfn "Epoch %i, minibatch %i, loss %f" j b (float32 L)
            b <- b + 1
            yield float32 L
        j <- j + 1|]

let createNetwork (l:int[]) =
    {layers = Array.init (l.Length - 1) (fun i ->
        {W = DM.init l.[i + 1] l.[i] (fun _ _ -> (-0.5 + rnd.NextDouble())/3.0 )
         b = DV.init l.[i + 1] (fun _ -> (-0.5 + rnd.NextDouble())/3.0 )
         a = sigmoid })}


let l1 = {
    W = DV.ReshapeToDM(2,DV [|0.5f;0.4f;0.3f;0.2f;0.1f;0.0f|]) |> DM.Transpose
    b = DV [|0.5f;0.4f;0.3f|]
    a = sigmoid
    }

let l2 = {
    W = [|[|-0.55f;-0.4f;-0.25f|]|] |> Array.map Array.toSeq |> Array.toSeq |> toDM
    b = DV [|-0.8f|]
    a = sigmoid
    }

let net1 = {layers = [|l1;l2|]}

let softmaxCrossEntropy (x:DM) (y:DM) =
    -(x |> DM.toCols |> Seq.mapi (fun i v -> 
        (DV.standardBasis v.Length (int (float32 y.[0, i]))) * log v) |> Seq.sum) / x.Cols

let logisticCrossEntropy (x:DM) (y:DM) =
    -((y .* (DM.Log x) + (1.0f-y) .* DM.Log (1.0f-x)) |> DM.Sum)

let squareSum (x:DM) (y:DM) =
    let r = x - y
    (DM.Pow(r,2) |> DM.Sum) / (2*y.Cols)

let XORx = [|[0.; 0.]
             [0.; 1.]
             [1.; 0.]
             [1.; 1.]
             |] |> Array.map List.toSeq |> Array.toSeq |> toDM |> DM.Transpose

let XORy = [|[0.0]
             [1.0]
             [1.0]
             [0.0]
             |] |> Array.map List.toSeq |> Array.toSeq |> toDM |> DM.Transpose

let tag = DiffSharp.Util.GlobalTagger.Next

l1.W <- l1.W |> makeReverse tag
l1.b <- l1.b |> makeReverse tag

l2.W <- l2.W |> makeReverse tag
l2.b <- l2.b |> makeReverse tag

let z_ = l1.W * XORx
let z1 = z_ + l1.b
let a1 = sigmoid(z1)

let z2 = l2.W * a1 + l2.b
let a2 = sigmoid(z2)

let log_a2 = DM.Log a2

let neg_target_plus_one = 1.0f-XORy

(*
// This one works correctly.
let neg_a2 = -a2
let neg_a2_plus_one = neg_a2+1.0f
// a2.A = DM [[0.30438143f; -1.4398216f; -1.53813016f; 0.297133237f]]
*)

// This one works incorrectly.
let neg_a2_plus_one = 1.0f - a2
// a2.A = DM [[-0.30438143f; -1.4398216f; -1.53813016f; -0.297133237f]]

let log_neg_a2_plus_one = DM.Log neg_a2_plus_one

let cross_entropy_left = XORy .* log_a2
let cross_entropy_right = neg_target_plus_one .* log_neg_a2_plus_one

let cross_entropy = cross_entropy_left + cross_entropy_right
let s = cross_entropy |> DM.Sum
let r = -0.25f*s

r |> reverseProp (D 1.0f)

