// Shit slow. Like 200x compared to my GPU code.

#I @"C:\F# Packages\packages\encog-dotnet-core.3.3.0\lib\net35"
#r "encog-core-cs.dll"
#load "load_mnist.fsx"

open Load_mnist.MnistLoad
open System
open Encog.Neural.Networks
open Encog.Neural.Networks.Layers
open Encog.Engine.Network.Activation
open Encog.ML.Data
open Encog.Neural.Networks.Training.Propagation.Resilient
open Encog.ML.Train
open Encog.ML.Data.Basic
open Encog

let train = make_imageset trainSetData trainSetLabels
let test = make_imageset testSetData testSetLabels

let MnistInput = 
    [|for i=0 to 299 do
        let t = i*784
        yield [|for j=t to 783+t do yield (float train.raw_data.[j])/255.0|]|]
let MnistOutput = 
    [|for i=0 to 299 do
        let r = Array.zeroCreate<float> 10
        r.[int train.raw_labels.[i]] <- 1.0
        yield r|]
let MnistValidationInput = 
    [|for i=0 to 9999 do
        let t = i*784
        yield [|for j=t to 783+t do yield (float test.raw_data.[j])/255.0|]|]
let MnistValidationOutput = 
    [|for i=0 to 9999 do
        let r = Array.zeroCreate<float> 10
        r.[int test.raw_labels.[i]] <- 1.0
        yield r|]

let network = new BasicNetwork()
network.AddLayer(new BasicLayer(null, true, 784));
network.AddLayer(new BasicLayer(new ActivationClippedLinear(), true, 1024));
network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 10));
network.Structure.FinalizeStructure();
network.Reset();


// create training data
let trainingSet = new BasicMLDataSet(MnistInput, MnistOutput);
let validationSet = new BasicMLDataSet(MnistValidationInput, MnistValidationOutput)

// train the neural network
let trainer = new ResilientPropagation(network, trainingSet);

let mutable epoch = 1;
let mutable f = true;

while f do
    trainer.Iteration();
    Console.WriteLine(@"Epoch #" + (string epoch) + @" Training error: " + (string trainer.Error));
    //let validationError = network.CalculateError(validationSet)
    //printfn "Validation Error: %f" validationError
    epoch <- epoch+1
    if epoch > 10 then f <- false



trainer.FinishTraining();
EncogFramework.Instance.Shutdown();