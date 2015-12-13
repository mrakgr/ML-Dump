open System
open System.Runtime.InteropServices
open FSharp.NativeInterop
open System.Security
open System.Threading.Tasks

#nowarn "9"
#nowarn "51"

type PinnedArray<'T when 'T : unmanaged> (array : 'T[]) =
    let h = GCHandle.Alloc(array, GCHandleType.Pinned)
    let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
    member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
    interface IDisposable with
        member this.Dispose() = h.Free()

type PinnedArray2D<'T when 'T : unmanaged> (array : 'T[,]) =
    let h = GCHandle.Alloc(array, GCHandleType.Pinned)
    let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
    member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
    interface IDisposable with
        member this.Dispose() = h.Free()

type LapackeOrder =
    | RowMajor = 101
    | ColMajor = 102

[<Literal>]
let blas_path = @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\DiffSharp.0.7.5\build\libopenblas.dll"

[<SuppressUnmanagedCodeSecurity>]
[<DllImport(blas_path)>]
extern int LAPACKE_sgesv(int order, int n, int nrhs, float32 *a, int lda, int *ipiv, float32 *b, int ldb)

// It works fine with col major, but throws an execution engine exception with row major. I have no idea idea what this is. Probably a library bug.
let sgesvc(a:float32[,], b:float32[]) =
    let m = Array2D.length1 a
    let n = Array2D.length2 a
    let a' = Array2D.copy a
    let b' = Array.copy b
    let ipiv = Array.zeroCreate n
    let mutable arg_order = LapackeOrder.ColMajor |> int
    let mutable arg_n = n
    let mutable arg_nrhs = 1
    let mutable arg_lda = n
    let mutable arg_ldb = n
    use arg_a = new PinnedArray2D<float32>(a')
    use arg_ipiv = new PinnedArray<int>(ipiv)
    use arg_b = new PinnedArray<float32>(b')
    let arg_info = LAPACKE_sgesv(arg_order, arg_n, arg_nrhs, arg_a.Ptr, arg_lda, arg_ipiv.Ptr, arg_b.Ptr, arg_ldb)
    printfn "arg_info = %i" arg_info
    if arg_info = 0 then
        Some(b')
    else
        None

let sgesvr(a:float32[,], b:float32[]) =
    let m = Array2D.length1 a
    let n = Array2D.length2 a
    let a' = Array2D.copy a
    let b' = Array.copy b
    let ipiv = Array.zeroCreate n
    let mutable arg_order = LapackeOrder.RowMajor |> int
    let mutable arg_n = n
    let mutable arg_nrhs = 1
    let mutable arg_lda = n
    let mutable arg_ldb = n
    use arg_a = new PinnedArray2D<float32>(a')
    use arg_ipiv = new PinnedArray<int>(ipiv)
    use arg_b = new PinnedArray<float32>(b')
    let arg_info = LAPACKE_sgesv(arg_order, arg_n, arg_nrhs, arg_a.Ptr, arg_lda, arg_ipiv.Ptr, arg_b.Ptr, arg_ldb)
    printfn "arg_info = %i" arg_info
    if arg_info = 0 then
        Some(b')
    else
        None

let rng = System.Random()

let a = Array2D.init 5 5 (fun _ _ -> rng.NextDouble() |> float32)
let b = Array.init 5 (fun _ -> rng.NextDouble() |> float32)

sgesvc(a,b) // Works fine.
sgesvr(a,b) // Crashes.