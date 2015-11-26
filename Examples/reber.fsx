// A function to generate Reber grammar strings.
// https://jamesmccaffrey.wordpress.com/2015/08/26/programmatically-generating-reber-grammar-strings/
// Is represented the graph in the above link.

type reberNode =
    | Node0 // Can only receive B. Outputs T or P.
    | Node1 // Can receive T or S. Outputs S or X.
    | Node2 // Can receive P or T. Outputs V or T.
    | Node3 // Can receive X or P. Outputs X or S.
    | Node4 // Can only receive V. Outputs P or V.
    | Node5 // Can receive S or V. Outputs E. This is the last node.

let rng = System.Random()

let rec make_random_reber_string str node =
    match node with
        | Node0 ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"T") Node1 else make_random_reber_string (str+"P") Node2
        | Node1 ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"S") Node1 else make_random_reber_string (str+"X") Node3
        | Node2 ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"T") Node2 else make_random_reber_string (str+"V") Node4
        | Node3 ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"X") Node2 else make_random_reber_string (str+"S") Node5
        | Node4 ->
            let p = rng.NextDouble()
            if p > 0.5 then make_random_reber_string (str+"P") Node3 else make_random_reber_string (str+"V") Node5
        | Node5 -> str+"E"

open System.Collections.Generic
let reber_set = new HashSet<string>()
for i=1 to 100000 do
    reber_set.Add (make_random_reber_string "B" Node0) |> ignore

let tens = reber_set |> Seq.filter (fun a -> a.Length <= 10)

tens |> Seq.iter (printfn "%s")
tens |> Seq.length
