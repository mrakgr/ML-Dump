(*
    let cudnn = CUDNN.Default

    let DataType = CUDNNInterop.cudnnDataType_t.CUDNN_DATA_FLOAT;
    let TensorFormat = CUDNNInterop.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW;

    let dstTensorDesc = new CUDNNTensorDescriptor()
    let srcTensorDesc = new CUDNNTensorDescriptor()
    let biasTensorDesc = new CUDNNTensorDescriptor()

    let activationForward2 activationType alpha (z1: dM) beta (dest: dM) =
        srcTensorDesc.Set4D(TensorFormat, DataType, 1, 1, z1.num_rows, z1.num_cols)
        dstTensorDesc.Set4D(TensorFormat, DataType, 1, 1, dest.num_rows, dest.num_cols)
        cudnn.ActivationForward(activationType, alpha, srcTensorDesc, z1.dArray.Ptr, beta, dstTensorDesc, dest.dArray.Ptr)
        dest

    let inline activationForward activationType alpha (z1: dM) =
        let dArray = worker.Malloc<float32>(z1.dArray.Length)
        let dest = {num_rows = z1.num_rows; num_cols = z1.num_cols; dArray = dArray}
        activationForward2 activationType alpha z1 0.0f dest

    let modeSigm = CUDNNInterop.cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID
    let modeRelu = CUDNNInterop.cudnnActivationMode_t.CUDNN_ACTIVATION_RELU
    let modeTanh = CUDNNInterop.cudnnActivationMode_t.CUDNN_ACTIVATION_TANH
    *)
    (*
    let activationGradient activationType (z1: dM) (diff1: dM) (dest: dM) (diff2: dM) =
            srcTensorDesc.Set4D(TensorFormat, DataType, 1, 1, z1.num_rows, z1.num_cols)
            dstTensorDesc.Set4D(TensorFormat, DataType, 1, 1, dest.num_rows, dest.num_cols)
            let alpha, beta = 1.f, 0.f
            cudnn.ActivationBackward(activationType, alpha, srcTensorDesc, z1.dArray.Ptr, srcTensorDesc, diff1.dArray.Ptr, dstTensorDesc, dest.dArray.Ptr, beta, dstTensorDesc, diff2.dArray.Ptr)
            dest, diff2

    let dest_z = createEmptyMatrixLike squared_cost_error
    let dest_diff = createEmptyMatrixLike z2

    let r1, r2 = activationGradient modeRelu z2  squared_cost_error dest_diff dest_z

    let e1 = squared_cost_error.dArray.Gather()
    let e2 = z2.dArray.Gather()

    let w1 = r1.dArray.Gather()
    let w2 = r2.dArray.Gather()
    w1 |> Array.min
    *)

    (*
    let addBias (weights: dM) (bias: dM) =
        dstTensorDesc.Set4D(TensorFormat, DataType, 1, weights.num_cols, weights.num_rows, 1)
        biasTensorDesc.Set4D(TensorFormat, DataType, 1, 1, bias.num_rows, bias.num_cols)
        let alpha, beta = 1.f, 1.f
        cudnn.AddTensor(CUDNNInterop.cudnnAddMode_t.CUDNN_ADD_IMAGE, alpha, biasTensorDesc, bias.dArray.Ptr, beta, dstTensorDesc, weights.dArray.Ptr)
    *)
