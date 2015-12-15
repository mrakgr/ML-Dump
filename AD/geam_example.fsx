#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/ManagedCuda.dll"
#r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaBlas.dll"

open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.VectorTypes
open ManagedCuda.CudaBlas

open System
open System.IO
open System.Collections

let ctx = new CudaContext()
let str = new CudaStream()
let cublas = CudaBlas(str.Stream)
let rng = System.Random()

let inline to_dev (host_ar: 't []) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

let inline to_dev' (host_ar: 't [,]) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
    let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
    dev_ar.CopyToHost(h_a)
    h_a

let inline new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
    new CudaDeviceVariable<'t>(SizeT n)

type dMatrix(num_rows:int,num_cols,dArray: CudaDeviceVariable<float32>) = 
    new(num_rows: int,num_cols) =
        let q = (num_rows*num_cols) |> SizeT
        let t = new CudaDeviceVariable<float32>(q)
        new dMatrix(num_rows,num_cols,t)

    new(num_rows: int,num_cols,dArray: float32[]) =
        let q = num_rows*num_cols
        if dArray.Length <> q then failwith "Invalid size in dMatrix construction."
        let t = to_dev dArray
        new dMatrix(num_rows,num_cols,t)

    member t.num_rows = num_rows
    member t.num_cols = num_cols
    member t.dArray = dArray

    override t.ToString() =
        sprintf "dM(%i,%i)" t.num_rows t.num_cols

    interface IDisposable with
        member t.Dispose() = dArray.Dispose()

let nT = Operation.Transpose
let T = Operation.NonTranspose

let t1 = Array.init 10 (fun _ -> 1.0f)
let t2 = Array.init 10 (fun _ -> 2.0f |> float32)
let t3 = Array.init 10 (fun _ -> 0.0f |> float32)
let d1 = new dMatrix(10,1,t1)
let d2 = new dMatrix(10,1,t2)
let d3 = new dMatrix(10,1,t3)

let m = 10 // Setting m to 2 and n to 4 causes it to throw the InvalidValue exception.
let n = 1
cublas.Geam(nT,nT,m,n,1.0f,d1.dArray,m,d2.dArray,m,1.0f,d3.dArray,m)
//cublas.Axpy(1.0f,d1.dArray,1,d3.dArray,1) //Axpy works fine as far as I can tell.
let alpha = to_dev [|1.0f|]
let beta = to_dev [|1.0f|]
CudaBlasNativeMethods.cublasSgeam(cublas.CublasHandle,nT,nT,m,n,alpha.DevicePointer,d1.dArray.DevicePointer,m,beta.DevicePointer,d2.dArray.DevicePointer,m,d3.dArray.DevicePointer,m)
// Pulling the native method seemed to have done nothing at first, but now I am seeing that it is causing it to crash completely with an AccessViolationException.
// My hypothesis is that it is passing the parameters incorrectly.
to_host d3.dArray

// Sgemm does not work either, but I've taken out the example for brevity. See ad_utils_spiral.fsx for the whole thing.