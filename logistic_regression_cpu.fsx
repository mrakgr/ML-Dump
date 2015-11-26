#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\MathNet.Numerics.FSharp.3.7.0\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\MathNet.Numerics.Data.Text.3.1.1\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\MathNet.Numerics.3.7.0\lib\net40\"
#r @"MathNet.Numerics.dll"
#r @"MathNet.Numerics.FSharp.dll"
#r @"MathNet.Numerics.Data.Text.dll"
#load "Types.fs"

open Mnist.Types
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Data.Text
open MathNet.Numerics.Distributions

open System
open System.IO

Control.NativeProviderPath <- @"C:\F# Packages\packages\MathNet.Numerics.MKL.Win-x64.1.8.0\content"
Control.UseNativeMKL()

type MnistMTX = {
    test_data : Matrix<float32>
    test_labels : Matrix<float32>
    train_data : Matrix<float32>
    train_labels : Matrix<float32>
    validation_data : Matrix<float32>
    validation_labels : Matrix<float32>
    }

let data_path = @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Load MNIST\Load MNIST\bin\Debug\"

//MatrixMarketWriter.WriteMatrix(data_path+"MnistTest_.mtx", , Compression.GZip)
    
let mnist_mtx = {
    test_data = 
        let q = MatrixMarketReader.ReadMatrix<float32>(data_path+"MnistTest_data.mtx", Compression.GZip)
        let t = DenseMatrix.create 1 q.ColumnCount 1.0f
        t.Stack(q)
    test_labels = MatrixMarketReader.ReadMatrix<float32>(data_path+"MnistTest_label.mtx", Compression.GZip)
    train_data =
        let q = MatrixMarketReader.ReadMatrix<float32>(data_path+"MnistTrain_data.mtx", Compression.GZip)
        let t = DenseMatrix.create 1 q.ColumnCount 1.0f
        t.Stack(q)
    train_labels = MatrixMarketReader.ReadMatrix<float32>(data_path+"MnistTrain_label.mtx", Compression.GZip)
    validation_data = 
        let q = MatrixMarketReader.ReadMatrix<float32>(data_path+"MnistValidation_data.mtx", Compression.GZip)
        let t = DenseMatrix.create 1 q.ColumnCount 1.0f
        t.Stack(q)
    validation_labels = MatrixMarketReader.ReadMatrix<float32>(data_path+"MnistValidation_label.mtx", Compression.GZip)
    }

let logistic z = 1.0f/(1.0f+exp(-z))

type LogisticRegParams = {
    batch_size : int
    num_trials : int
    num_iterations_per_trial : int
    learning_rate: float32
    lambda : float32
    }

let logistic_test =
    for i in {-89.0f..0.1f..17.0f} do
        let t = logistic i
        let r1 = log(t)
        if Single.IsNaN(r1) || Single.IsInfinity(r1) then
            printfn "%f %f" i r1
        let r2 = log(1.0f - t)
        if Single.IsNaN(r2) || Single.IsInfinity(r2) then
            printfn "%f %f" i r2

let main =
    let logistic_regression (r : MnistMTX) (p: LogisticRegParams) =
        let batch_start = 0
        let batch_end = p.batch_size-1

        let dist = ContinuousUniform(-1e-3,1e-3)
        let mutable weights = DenseMatrix.random<float32> mnist_mtx.train_data.RowCount 10 dist

        let calculate_cost (data_batch: Matrix<float32>) (label_batch: Matrix<float32>) (weights: Matrix<float32>) = 
            let float_num_batches = float32 data_batch.ColumnCount
            let output = weights.TransposeThisAndMultiply(data_batch).Map(fun z -> logistic z)
            let log_output = output.PointwiseLog()
            let log_output2 = output.Map(fun x -> log(1.0f - x))
            let cross_entropy_cost = (-1.0f/float_num_batches)*((label_batch.*log_output+(1.0f-label_batch).*log_output2).ColumnSums().Sum())
            let reg_cost = (p.lambda/float_num_batches)*(weights.PointwisePower(2.0f).RowSums().Sum())
            let weights_grad = data_batch.TransposeAndMultiply((output-label_batch)/float_num_batches)
            let weights_reg = (2.0f*p.lambda/float_num_batches)*weights
            cross_entropy_cost+reg_cost, weights_grad+weights_reg

        let rec training_loop batch_start batch_end iter weights =
            let data_batch = mnist_mtx.train_data.[0..,batch_start..batch_end]
            let label_batch = mnist_mtx.train_labels.[0..,batch_start..batch_end]

            let rec gradient_descent_logistic_regression (data_batch: Matrix<float32>) (label_batch: Matrix<float32>) (weights: Matrix<float32>) =
                let cost, grad = calculate_cost data_batch label_batch weights
                let v = weights - p.learning_rate*grad
                cost, v

            let cost, new_weights = gradient_descent_logistic_regression data_batch label_batch weights
            //if iter % 10 = 0 then printfn "Cost in iteration %i is %f..." iter cost
            if iter < p.num_iterations_per_trial then
                let batch_start_next = batch_start+p.batch_size
                let batch_end_next = batch_end+p.batch_size
                if batch_end_next <= r.train_data.ColumnCount-1 then
                    training_loop batch_start_next batch_end_next (iter+1) new_weights
                else
                    training_loop 0 (p.batch_size-1) (iter+1) new_weights
            else
                new_weights

        for trial=1 to p.num_trials do
            weights <- training_loop batch_start batch_end 1 weights
            let validation_cost, _ = calculate_cost r.validation_data r.validation_labels weights
            //let test_cost, _ = calculate_cost r.validation_data r.validation_labels weights
            printfn "-----"
            printfn "The cross entropy cost on the validation set at trial %i is %f" trial validation_cost
            //printfn "DEBUG: The sum of all the weights is %f" (weights.RowSums().Sum())
            if Single.IsNaN(validation_cost) then failwith "Cost should not be Nan." 
            //printfn "The cross entropy cost on the test set at trial %i is %f" trial test_cost
        weights
        
    logistic_regression mnist_mtx {batch_size=250; num_trials=200; num_iterations_per_trial=300; learning_rate=0.2f; lambda = 0.1f}

let get_accuracy_log_reg (r : MnistMTX) (weights: Matrix<float32>) =
    let pred_matrix = weights.TransposeThisAndMultiply(r.test_data)
    let predictions = [|for x in pred_matrix.EnumerateColumns() -> x.MaximumIndex()|]
    let ground_truths = [|for x in mnist_mtx.test_labels.EnumerateColumns() -> x.MaximumIndex()|]
    let hits_array = Array.map2 (fun x y -> if x = y then 1.0f else 0.0f) predictions ground_truths
    let hits = hits_array |> Array.sum
    hits / float32 hits_array.Length

get_accuracy_log_reg mnist_mtx main
