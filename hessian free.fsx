let t1 = [|false;true|]
let t2 = [|false;true|]
let t3 = [|false;true|]
for i in t1 do
    for j in t2 do
        for k in t3 do
            let w1 = i || (j && k)
            let w2 = (i || j) && (i || k)
            printfn "%b, %b" w1 w2
