// A failed attempt at redesign. In the middle of this I realized that what I am doing would not improve on the previous
// class at all.
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

let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;

type DataTensor(data: d4M) =
    inherit DisposableObject()

    let srcTensorDesc = new CUDNNTensorDescriptor()
    do srcTensorDesc.Set4D(TensorFormat, DataType, data.num_feature_maps, data.num_channels, data.num_rows, data.num_cols)

    member this.nchw = data.num_feature_maps, data.num_channels, data.num_rows, data.num_cols
    member this.desc = srcTensorDesc
    member this.data = data
    member this.eq (r: DataTensor) =
        let n,c,h,w = this.nchw
        let n2,c2,h2,w2 = r.nchw

        if n <> n2 then failwith "n <> n2 in convolutionForward"
        if c <> c2 then failwith "c <> c2 in convolutionForward"
        if h <> h2 then failwith "h <> h2 in convolutionForward"
        if w <> w2 then failwith "w <> w2 in convolutionForward"

    override net.Dispose(disposing:bool) =
        if disposing then
            srcTensorDesc.Dispose()
            data.dArray.Dispose()

type FilterTensor(convolutional_filters: d4M) =
    inherit DisposableObject()

    let filterDesc = new CUDNNFilterDescriptor()
    do filterDesc.Set4D(DataType, convolutional_filters.num_feature_maps, convolutional_filters.num_channels, convolutional_filters.num_rows, convolutional_filters.num_cols)

    member this.desc = filterDesc
    member this.filter = convolutional_filters

    new(n,c,h,w,scale) = new FilterTensor(createRandomUniform4DMatrix n c h w scale)

    override net.Dispose(disposing:bool) =
        if disposing then
            filterDesc.Dispose()
            convolutional_filters.dArray.Dispose()

let cross_correlation_mode = CUDNNInterop.cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION
let convolution_mode = CUDNNInterop.cudnnConvolutionMode_t.CUDNN_CONVOLUTION
type ConvolutionalDescriptor(pad_h, pad_w, u, v, upscalex, upscaley, mode) =
    inherit DisposableObject()

    let convDesc = new CUDNNConvolutionDescriptor()
    do convDesc.Set2D(pad_h, pad_w, u, v, upscalex, upscaley, mode)

    new() = new ConvolutionalDescriptor(0, 0, 1, 1, 1, 1, CUDNNInterop.cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION)

    member this.desc = convDesc

    override net.Dispose(disposing:bool) =
        if disposing then
            convDesc.Dispose()

type ConvLayer(input_tensor: DataTensor, filter_tensor: FilterTensor, convolutional_desc: ConvolutionalDescriptor) =
    inherit DisposableObject()
    
    // find dimension of convoltion output
    // outputDim = 1 + (inputDim + 2*pad - filterDim) / convolutionStride
    let output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w = convolutional_desc.desc.Get2DForwardOutputDim(input_tensor.desc, filter_tensor.desc)
    
    let output_tensor = new DataTensor(createEmpty4DMatrix output_matrix_n output_matrix_c output_matrix_h output_matrix_w)
    let error_tensor = new DataTensor(createEmpty4DMatrix output_matrix_n output_matrix_c output_matrix_h output_matrix_w)

    let algo = cudnn.GetConvolutionForwardAlgorithm(input_tensor.desc, filter_tensor.desc, convolutional_desc.desc, output_tensor.desc, CUDNNInterop.cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, IntPtr 0)
    let sizeInBytes = cudnn.GetConvolutionForwardWorkspaceSize(input_tensor.desc, filter_tensor.desc, convolutional_desc.desc, output_tensor.desc, algo)
    let workSpace = worker.Malloc<byte>(sizeInBytes.ToInt32())

    let gradient_matrix = createEmpty4DMatrixLike filter_tensor.filter
    do saxpy2 0.0f gradient_matrix gradient_matrix
    let gradient_filter = new FilterTensor(gradient_matrix)

    override net.Dispose(disposing:bool) =
        if disposing then
            input_tensor.Dispose()
            filter_tensor.Dispose()
            output_tensor.Dispose()
            error_tensor.Dispose()
            gradient_filter.Dispose()
            poolingDesc.Dispose()
            workSpace.Dispose()
            convolutional_filters.dArray.Dispose()
            output_matrix.dArray.Dispose()
            error_matrix.dArray.Dispose()
            gradient_matrix.dArray.Dispose()

    member this.getSourceData = srcTensorDesc
    member this.getErrorData = dstTensorDesc, error_matrix
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
        (*
        let n, c, h, w = batch_sample.num_feature_maps, batch_sample.num_channels, batch_sample.num_rows, batch_sample.num_cols
        if n <> input.num_feature_maps then failwith "n <> input.num_feature_maps in activationBackward"
        if c <> input.num_channels then failwith "c <> input.num_channels in activationBackward"
        if h <> input.num_rows then failwith "h <> input.num_rows in activationBackward"
        if w <> input.num_cols then failwith "w <> input.num_cols in activationBackward"
        *)

        cudnn.ActivationBackward(activation_type,alpha,dstTensorDesc,output.dArray.Ptr,dstTensorDesc,error.dArray.Ptr,dstTensorDesc,output.dArray.Ptr,beta,dstTensorDesc,activated_error.dArray.Ptr)

    member this.convolutionBackwardData(alpha, error_above: d4M, beta, error_below_dsc, error_below: d4M) =
        let n, c, h, w = output_matrix_n,output_matrix_c,output_matrix_h,output_matrix_w
        if n <> error_above.num_feature_maps then failwith "n <> error_above.num_feature_maps in convolutionBackwardData"
        if c <> error_above.num_channels then failwith "c <> error_above.num_channels in convolutionBackwardData"
        if h <> error_above.num_rows then failwith "h <> error_above.num_rows in convolutionBackwardData"
        if w <> error_above.num_cols then failwith "w <> error_above.num_cols in convolutionBackwardData"

        cudnn.ConvolutionBackwardData(alpha,filterDesc,convolutional_filters.dArray.Ptr,dstTensorDesc,error_above.dArray.Ptr,convDesc,beta,error_below_dsc,error_below.dArray.Ptr)

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


