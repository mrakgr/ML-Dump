// To see how the DeviceUnaryTransformModule looks like in Alea take a peek in ad_utils_v3.fsx.

#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\ManagedCuda.dll"
#r @"C:\Users\Marko\documents\visual studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\NVRTC.dll"

open ManagedCuda
open ManagedCuda.VectorTypes
open ManagedCuda.BasicTypes
open ManagedCuda.NVRTC

let to_dev (host_ar: 't []) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

let to_host (dev_ar: CudaDeviceVariable<'t>) =
    let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
    dev_ar.CopyToHost(h_a)
    h_a

let new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
    new CudaDeviceVariable<'t>(SizeT n)

let ctx = new CudaContext()

/// Unary transform module for applying single functions to an array.
type DeviceUnaryTransformModule(op: string) = 
    let block_size = 128

    let kernel_code = "
        //Kernel code:
        extern \"C\" {
            __device__ inline float op(float x)
            {
                return "+op+"
            }
        
            // Device code
            __global__ void Map1Kernel(const float* A, float* O, int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i < N)
                    O[i] = op(A[i]);
            }
        }

        "
    let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map1Kernel")
    do  
        try k.Compile([||])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())

    let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map1Kernel")

    member t.A(x: CudaDeviceVariable<float32>) =
        let n = int x.Size
        let o = new_dev<float32> n
        kernel.GridDimensions <- dim3((n+block_size-1)/block_size)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.Run(x.DevicePointer,o.DevicePointer,n) |> ignore
        o

    member t.A(x: CudaDeviceVariable<float32>, o: CudaDeviceVariable<float32>) =
        let n = int o.Size
        kernel.GridDimensions <- dim3((n+block_size-1)/block_size)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.Run(x.DevicePointer,o.DevicePointer,n) |> ignore

let rng = System.Random()

let n = 100
let h_a = Array.init n (fun _ -> (rng.NextDouble()-0.5)*6.0 |> float32)
let d_a = to_dev h_a

let sigmoidModule = DeviceUnaryTransformModule "1.0f / (1.0f + expf(-x));"

let d_o_sigmoid = sigmoidModule.A(d_a)
let h_o_sigmoid = to_host d_o_sigmoid

let t_o_sigmoid = h_a |> Array.map(fun x -> 1.0f / (1.0f + exp(-x)))

let sigmoid_diffs = Array.map2(fun a b -> a-b) t_o_sigmoid h_o_sigmoid
let sigmoid_diffs_sum = sigmoid_diffs |> Array.sum

// Cuda math functions ref: http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
let tanhModule = DeviceUnaryTransformModule "tanhf(x);"

let d_o_tanh = tanhModule.A(d_a)
let h_o_tanh = to_host d_o_tanh

let t_o_tanh = h_a |> Array.map(fun x -> tanh(x))

let tanh_diffs = Array.map2(fun a b -> a-b) t_o_tanh h_o_tanh
let tanh_diffs_sum = tanh_diffs |> Array.sum

let reluModule = DeviceUnaryTransformModule "x > 0.0f ? x : 0.0f;"

let d_o_relu = reluModule.A(d_a)
let h_o_relu = to_host d_o_relu

let t_o_relu = h_a |> Array.map(fun x -> if x > 0.0f then x else 0.0f)

let relu_diffs = Array.map2(fun a b -> a-b) t_o_relu h_o_relu
let relu_diffs_sum = relu_diffs |> Array.sum