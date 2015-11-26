// As I am starting with convolutional nets, I am going to build the infrastructure for it here.
#load "load_mnist.fsx"
open Load_mnist.MnistLoad
#load "utils.fsx"
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

let cudnn = new CUDNN(worker)

// For errors without activations.
let binaryErrorModule = new DeviceBinaryTransformModule<float32> <@ fun y a -> a-y @>

let sigmoid_act = cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID
let relu_act = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU

let saxpy2 (alpha: float32) (x: d4M) (y: d4M) =
    if x.dArray.Length <> y.dArray.Length then failwith "x.dArray.Length <> y.dArray.Length in saxpy2"
    if x.num_feature_maps <> y.num_feature_maps then failwith "x.num_feature_maps <> y.num_feature_maps in activationBackward"
    if x.num_channels <> y.num_channels then failwith "x.num_channels <> y.num_channels in activationBackward"
    if x.num_rows <> y.num_rows then failwith "x.num_rows <> y.num_rows in activationBackward"
    if x.num_cols <> y.num_cols then failwith "x.num_cols <> y.num_cols in activationBackward"
    cublas.Saxpy(x.dArray.Length,alpha,x.dArray.Ptr,1,y.dArray.Ptr,1)

type ConvolutionParameters = {
    pad_h : int
    pad_w : int
    stride_h : int
    stride_w : int
    upscale_h : int
    upscale_w : int
    mode : cudnnConvolutionMode_t
    }

type ConvLayer(batch_sample: d4M, convolutional_filters: d4M, convPar: ConvolutionParameters) =
    inherit DisposableObject()
    
    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;

    let srcTensorDesc = new CUDNNTensorDescriptor()
    let dstTensorDesc = new CUDNNTensorDescriptor()
    let filterDesc = new CUDNNFilterDescriptor()
    let convDesc = new CUDNNConvolutionDescriptor()
    
    do srcTensorDesc.Set4D(TensorFormat, DataType, batch_sample.num_feature_maps, batch_sample.num_channels, batch_sample.num_rows, batch_sample.num_cols)
    do filterDesc.Set4D(DataType, convolutional_filters.num_feature_maps, convolutional_filters.num_channels, convolutional_filters.num_rows, convolutional_filters.num_cols)

    // CUDNN_CROSS_CORRELATION is the transpose of CUDNN_CONVOLUTION.
    do convDesc.Set2D(convPar.pad_h, convPar.pad_w, convPar.stride_h, convPar.stride_w, convPar.upscale_h, convPar.upscale_w, convPar.mode)

    // find dimension of convoltion output
    // outputDim = 1 + (inputDim + 2*pad - filterDim) / convolutionStride
    let output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w = convDesc.Get2DForwardOutputDim(srcTensorDesc, filterDesc)
    
    let output_matrix = createEmpty4DMatrix output_matrix_n output_matrix_c output_matrix_h output_matrix_w
    let error_matrix = createEmpty4DMatrix output_matrix_n output_matrix_c output_matrix_h output_matrix_w

    do dstTensorDesc.Set4D(TensorFormat, DataType, output_matrix_n, output_matrix_c, output_matrix_h, output_matrix_w)
    let algo = cudnn.GetConvolutionForwardAlgorithm(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, CUDNNInterop.cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, IntPtr 0)
    let sizeInBytes = cudnn.GetConvolutionForwardWorkspaceSize(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo)
    let workSpace = worker.Malloc<byte>(sizeInBytes.ToInt32())

    let gradient_matrix = createEmpty4DMatrixLike convolutional_filters
    do saxpy2 0.0f gradient_matrix gradient_matrix

    /// Creates random filters
    new(batch_sample, filters) = 
        let default_parameters = {
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            upscale_h = 1
            upscale_w = 1
            mode = cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION
            }
        new ConvLayer(batch_sample, filters,default_parameters)

    override net.Dispose(disposing:bool) =
        if disposing then
            srcTensorDesc.Dispose()
            dstTensorDesc.Dispose()
            filterDesc.Dispose()
            convDesc.Dispose()
            workSpace.Dispose()
            convolutional_filters.dArray.Dispose()
            output_matrix.dArray.Dispose()
            error_matrix.dArray.Dispose()
            gradient_matrix.dArray.Dispose()

    member this.getSourceData = srcTensorDesc
    member this.getErrorMatrix = error_matrix
    member this.getGradientMatrix = gradient_matrix

    member this.convolutionForward(alpha, batch: d4M, beta) =
        let n, c, h, w = batch_sample.num_feature_maps, batch_sample.num_channels, batch_sample.num_rows, batch_sample.num_cols
        if n <> batch.num_feature_maps then failwith "n <> batch.num_feature_maps in convolutionForward"
        if c <> batch.num_channels then failwith "c <> batch.num_channels in convolutionForward"
        if h <> batch.num_rows then failwith "h <> batch.num_rows in convolutionForward"
        if w <> batch.num_cols then failwith "w <> batch.num_cols in convolutionForward"

        cudnn.ConvolutionForward(alpha, srcTensorDesc, batch.dArray.Ptr, filterDesc, convolutional_filters.dArray.Ptr, convDesc, algo, workSpace.Ptr, sizeInBytes, beta, dstTensorDesc, output_matrix.dArray.Ptr)
        output_matrix

    // This is different
    member this.convolutionBackwardFilter(alpha, prev_layer_activations: d4M, error: d4M,  beta) =
        let n, c, h, w = batch_sample.num_feature_maps, batch_sample.num_channels, batch_sample.num_rows, batch_sample.num_cols
        if n <> prev_layer_activations.num_feature_maps then failwith "n <> prev_layer_activations.num_feature_maps in convolutionBackwardFilter"
        if c <> prev_layer_activations.num_channels then failwith "c <> prev_layer_activations.num_channels in convolutionBackwardFilter"
        if h <> prev_layer_activations.num_rows then failwith "h <> prev_layer_activations.num_rows in convolutionBackwardFilter"
        if w <> prev_layer_activations.num_cols then failwith "w <> prev_layer_activations.num_cols in convolutionBackwardFilter"

        let n, c, h, w = output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w
        if n <> error.num_feature_maps then failwith "n <> error.num_feature_maps in convolutionBackwardFilter"
        if c <> error.num_channels then failwith "c <> error.num_channels in convolutionBackwardFilter"
        if h <> error.num_rows then failwith "h <> error.num_rows in convolutionBackwardFilter"
        if w <> error.num_cols then failwith "w <> error.num_cols in convolutionBackwardFilter"

        cudnn.ConvolutionBackwardFilter(alpha, srcTensorDesc, prev_layer_activations.dArray.Ptr,dstTensorDesc,error.dArray.Ptr,convDesc,beta,filterDesc,gradient_matrix.dArray.Ptr)
        gradient_matrix

    /// output = f(input)
    /// Sigmoid mode will read from the output, while Relu will require input.
    member this.activationBackward(alpha, output: d4M, error: d4M, beta, activated_error: d4M, activation_type) =
        let n, c, h, w = output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w
        if n <> output.num_feature_maps then failwith "n <> output.num_feature_maps in activationBackward"
        if c <> output.num_channels then failwith "c <> output.num_channels in activationBackward"
        if h <> output.num_rows then failwith "h <> output.num_rows in activationBackward"
        if w <> output.num_cols then failwith "w <> output.num_cols in activationBackward"

        if n <> error.num_feature_maps then failwith "n <> error.num_feature_maps in activationBackward"
        if c <> error.num_channels then failwith "c <> error.num_channels in activationBackward"
        if h <> error.num_rows then failwith "h <> error.num_rows in activationBackward"
        if w <> error.num_cols then failwith "w <> error.num_cols in activationBackward"

        cudnn.ActivationBackward(activation_type,alpha,dstTensorDesc,output.dArray.Ptr,dstTensorDesc,error.dArray.Ptr,dstTensorDesc,output.dArray.Ptr,beta,dstTensorDesc,activated_error.dArray.Ptr)

    member this.convolutionBackwardData(alpha, error_above: d4M, beta, error_below: d4M) =
        let n, c, h, w = output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w
        if n <> error_above.num_feature_maps then failwith "n <> error_above.num_feature_maps in convolutionBackwardData"
        if c <> error_above.num_channels then failwith "c <> error_above.num_channels in convolutionBackwardData"
        if h <> error_above.num_rows then failwith "h <> error_above.num_rows in convolutionBackwardData"
        if w <> error_above.num_cols then failwith "w <> error_above.num_cols in convolutionBackwardData"

        cudnn.ConvolutionBackwardData(alpha,filterDesc,convolutional_filters.dArray.Ptr,dstTensorDesc,error_above.dArray.Ptr,convDesc,beta,srcTensorDesc,error_below.dArray.Ptr)

    // -(labels-output) = output-labels
    member this.lastLayerError(output: d4M, labels: d4M) =
            let n, c, h, w = output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w
            if n <> output.num_feature_maps then failwith "n <> output.num_feature_maps in activationBackward"
            if c <> output.num_channels then failwith "c <> output.num_channels in activationBackward"
            if h <> output.num_rows then failwith "h <> output.num_rows in activationBackward"
            if w <> output.num_cols then failwith "w <> output.num_cols in activationBackward"

            if n <> error_matrix.num_feature_maps then failwith "n <> error.num_feature_maps in activationBackward"
            if c <> error_matrix.num_channels then failwith "c <> error.num_channels in activationBackward"
            if h <> error_matrix.num_rows then failwith "h <> error.num_rows in activationBackward"
            if w <> error_matrix.num_cols then failwith "w <> error.num_cols in activationBackward"

            if n <> labels.num_feature_maps then failwith "n <> labels.num_feature_maps in activationBackward"
            if c <> labels.num_channels then failwith "c <> labels.num_channels in activationBackward"
            if h <> labels.num_rows then failwith "h <> labels.num_rows in activationBackward"
            if w <> labels.num_cols then failwith "w <> labels.num_cols in activationBackward"

            binaryErrorModule.Apply(error_matrix.dArray.Length,labels.dArray.Ptr,output.dArray.Ptr,error_matrix.dArray.Ptr)
            error_matrix

    member this.ActivationForward(alpha, input: d4M, beta, output: d4M, activation_type) =
        let n, c, h, w = output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w
        if n <> output.num_feature_maps then failwith "n <> output.num_feature_maps in activationBackward"
        if c <> output.num_channels then failwith "c <> output.num_channels in activationBackward"
        if h <> output.num_rows then failwith "h <> output.num_rows in activationBackward"
        if w <> output.num_cols then failwith "w <> output.num_cols in activationBackward"

        if n <> input.num_feature_maps then failwith "n <> input.num_feature_maps in activationBackward"
        if c <> input.num_channels then failwith "c <> input.num_channels in activationBackward"
        if h <> input.num_rows then failwith "h <> input.num_rows in activationBackward"
        if w <> input.num_cols then failwith "w <> input.num_cols in activationBackward"

        cudnn.ActivationForward(activation_type,alpha,dstTensorDesc,input.dArray.Ptr,beta,dstTensorDesc,output.dArray.Ptr)

let max_pooling = cudnnPoolingMode_t.CUDNN_POOLING_MAX
type PoolingLayer(input_sample: d4M, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride, create_error_matrix: bool) =
    inherit DisposableObject()
    
    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;

    let srcTensorDesc = new CUDNNTensorDescriptor()
    let dstTensorDesc = new CUDNNTensorDescriptor()
    let poolingDesc = new CUDNNPoolingDescriptor()

    do srcTensorDesc.Set4D(TensorFormat, DataType, input_sample.num_feature_maps, input_sample.num_channels, input_sample.num_rows, input_sample.num_cols)
    do poolingDesc.Set2D(mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride)

    // find dimension of convoltion output
    // outputDim = 1 + (inputDim + 2*pad - filterDim) / convolutionStride
    let output_matrix_n = input_sample.num_feature_maps
    let output_matrix_c = input_sample.num_channels
    let output_matrix_h = 1 + (input_sample.num_rows + 2*verticalPadding - windowHeight) / verticalStride
    let output_matrix_w = 1 + (input_sample.num_cols + 2*horizontalPadding - windowWidth) / horizontalStride

    let output_matrix = createEmpty4DMatrix output_matrix_n output_matrix_c output_matrix_h output_matrix_w
    let error_matrix = 
        if create_error_matrix then createEmpty4DMatrix output_matrix_n output_matrix_c output_matrix_h output_matrix_w
        else createEmpty4DMatrix 0 0 0 0

    do dstTensorDesc.Set4D(TensorFormat, DataType, output_matrix_n, output_matrix_c, output_matrix_h, output_matrix_w)

    new(input_sample: d4M, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride) =
        new PoolingLayer(input_sample, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride, true)

    member this.getSourceData = srcTensorDesc
    member this.getErrorMatrix = error_matrix

    member this.poolingForward(alpha, batch: d4M, beta) =
        let n, c, h, w = input_sample.num_feature_maps, input_sample.num_channels, input_sample.num_rows, input_sample.num_cols
        if n <> batch.num_feature_maps then failwith "n <> batch.num_feature_maps in poolingForward"
        if c <> batch.num_channels then failwith "c <> batch.num_channels in poolingForward"
        if h <> batch.num_rows then failwith "h <> batch.num_rows in poolingForward"
        if w <> batch.num_cols then failwith "w <> batch.num_cols in poolingForward"

        cudnn.PoolingForward(poolingDesc,alpha,srcTensorDesc,batch.dArray.Ptr,beta,dstTensorDesc,output_matrix.dArray.Ptr)
        output_matrix

    member this.poolingBackward(alpha: float32, output, err, input, beta, err_below) =
        let n, c, h, w = input_sample.num_feature_maps, input_sample.num_channels, input_sample.num_rows, input_sample.num_cols
        if n <> input.num_feature_maps then failwith "n <> input.num_feature_maps in poolingBackward"
        if c <> input.num_channels then failwith "c <> input.num_channels in poolingBackward"
        if h <> input.num_rows then failwith "h <> input.num_rows in poolingBackward"
        if w <> input.num_cols then failwith "w <> input.num_cols in poolingBackward"

        if n <> err_below.num_feature_maps then failwith "n <> err_below.num_feature_maps in poolingBackward"
        if c <> err_below.num_channels then failwith "c <> err_below.num_channels in poolingBackward"
        if h <> err_below.num_rows then failwith "h <> err_below.num_rows in poolingBackward"
        if w <> err_below.num_cols then failwith "w <> err_below.num_cols in poolingBackward"

        let n, c, h, w = output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w
        if n <> output.num_feature_maps then failwith "n <> output.num_feature_maps in poolingBackward"
        if c <> output.num_channels then failwith "c <> output.num_channels in poolingBackward"
        if h <> output.num_rows then failwith "h <> output.num_rows in poolingBackward"
        if w <> output.num_cols then failwith "w <> output.num_cols in poolingBackward"

        if n <> err.num_feature_maps then failwith "n <> err.num_feature_maps in poolingBackward"
        if c <> err.num_channels then failwith "c <> err.num_channels in poolingBackward"
        if h <> err.num_rows then failwith "h <> err.num_rows in poolingBackward"
        if w <> err.num_cols then failwith "w <> err.num_cols in poolingBackward"

        cudnn.PoolingBackward(poolingDesc,alpha,dstTensorDesc,output.dArray.Ptr,dstTensorDesc,err.dArray.Ptr,srcTensorDesc,input.dArray.Ptr,beta,srcTensorDesc,err_below.dArray.Ptr)
        
    override net.Dispose(disposing:bool) =
        if disposing then
            srcTensorDesc.Dispose()
            dstTensorDesc.Dispose()
            poolingDesc.Dispose()
            output_matrix.dArray.Dispose()
            error_matrix.dArray.Dispose()
