#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.2.1.2.3274\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.IL.2.1.2.3274\lib\net40\"
#I @"C:\Users\Marko\documents\visual studio 2015\Projects\Load MNIST\packages\Alea.CUDA.Unbound.2.1.2.3274\lib\net40\"
#r @"Alea.CUDA.Unbound.dll"
#r @"Alea.CUDA.IL.dll"
#r @"Alea.CUDA.dll"
#r "System.Configuration.dll"

module Transform = 
    open System
    open Alea.CUDA
    open Alea.CUDA.Utilities
    open Alea.CUDA.CULib
    open Alea.CUDA.IL
    open FSharp.Quotations

    open Alea.CUDA.Unbound.Rng
    open Alea.CUDA.Unbound

    
    type dMatrix<'T> = {
        num_rows : int
        num_cols : int
        dArray : DeviceMemory<'T>
    }

    Alea.CUDA.Settings.Instance.Resource.AssemblyPath <- __SOURCE_DIRECTORY__ + @"\..\..\..\packages\Alea.CUDA.2.0.3057\private"
    Alea.CUDA.Settings.Instance.Resource.Path <- __SOURCE_DIRECTORY__ + @"\..\..\..\release"

    type BinaryTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T>) =
        inherit GPUModule(target)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (z:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                z.[i] <- __eval(op) x.[i] y.[i]
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y z

        member this.Apply(x:'T[], y:'T[]) =
            let n = x.Length
            use x = this.GPUWorker.Malloc(x)
            use y = this.GPUWorker.Malloc(y)
            use z = this.GPUWorker.Malloc(n)
            this.Apply(n, x.Ptr, y.Ptr, z.Ptr)
            z.Gather()

    type DeviceBinaryTransformModule<'T>(target, op:Expr<'T -> 'T -> 'T>) =
        inherit GPUModule(target)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (z:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                z.[i] <- __eval(op) x.[i] y.[i]
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>, z:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y z

        member this.Apply (x: DeviceMemory<'T>, y: DeviceMemory<'T>) =
            let z = this.GPUWorker.Malloc(x.Length)
            this.Apply(x.Length, x.Ptr, y.Ptr, z.Ptr)
            z

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>) =
            let z = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = z}

        member this.Apply (x: DeviceMemory<'T>, y: DeviceMemory<'T>, z: DeviceMemory<'T>) =
            this.Apply(x.Length, x.Ptr, y.Ptr, z.Ptr)
            z

        member this.Apply (x: dMatrix<'T>, y: dMatrix<'T>, z: dMatrix<'T>) =
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.dArray.Ptr, z.dArray.Ptr)
            z

    let generate n =
        let rng = Random()
        let x = Array.init n (fun _ -> rng.NextDouble())
        let y = Array.init n (fun _ -> rng.NextDouble())
        x, y

    type UnaryTransformModule<'T>(target, op:Expr<'T -> 'T>) =
        inherit ILGPUModule(target)

        new (target, op:Func<'T, 'T>) =
            new UnaryTransformModule<'T>(target, <@ fun x -> op.Invoke(x) @>)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                y.[i] <- __eval(op) x.[i] 
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y 

        member this.Apply (x:'T[]) =
            use x = this.GPUWorker.Malloc(x)
            use y = this.GPUWorker.Malloc(x.Length)
            this.Apply(x.Length, x.Ptr, y.Ptr)
            y.Gather()

    type UnaryDeviceTransformModule<'T>(target, op:Expr<'T -> 'T>) =
        inherit ILGPUModule(target)

        new (target, op:Func<'T, 'T>) =
            new UnaryDeviceTransformModule<'T>(target, <@ fun x -> op.Invoke(x) @>)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) (y:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                y.[i] <- __eval(op) x.[i] 
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>, y:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x y

        member this.Apply (x: DeviceMemory<'T>) =
            let y = this.GPUWorker.Malloc(x.Length)
            this.Apply(x.Length, x.Ptr, y.Ptr)
            y

        member this.Apply (x: dMatrix<'T>) =
            let y = this.GPUWorker.Malloc(x.dArray.Length)
            this.Apply(x.dArray.Length, x.dArray.Ptr, y.Ptr)
            {num_rows = x.num_rows; num_cols = x.num_cols; dArray = y}

    type UnaryDeviceInPlaceTransformModule<'T>(target, op:Expr<'T -> 'T>) =
        inherit ILGPUModule(target)

        new (target, op:Func<'T, 'T>) =
            new UnaryDeviceInPlaceTransformModule<'T>(target, <@ fun x -> op.Invoke(x) @>)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel (n:int) (x:deviceptr<'T>) =
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start 
            while i < n do
                x.[i] <- __eval(op) x.[i] 
                i <- i + stride

        member this.Apply(n:int, x:deviceptr<'T>) =
            let blockSize = 256
            let numSm = this.GPUWorker.Device.Attributes.MULTIPROCESSOR_COUNT
            let gridSize = min (16 * numSm) (divup n blockSize)
            let lp = LaunchParam(gridSize, blockSize)
            this.GPULaunch <@ this.Kernel @> lp n x

        member this.Apply (x: DeviceMemory<'T>) =
            this.Apply(x.Length, x.Ptr)
            x

        member this.Apply (x: dMatrix<'T>) =
            this.Apply(x.dArray.Length, x.dArray.Ptr)
            x

    type rowReduceModule<'T when 'T : comparison>(target, rows : int) =
        inherit ILGPUModule(target)

        [<Kernel;ReflectedDefinition>]
        member this.Kernel cols (x:deviceptr<'T>) (y:deviceptr<int>) =
            let tid = blockIdx.x * blockDim.x + threadIdx.x
            let start = tid*rows
            let mutable cost, row_index = x.[start], 0
            if tid < cols then
                for i=1 to rows-1 do
                    let new_cost = x.[start+i]
                    if cost < new_cost then
                        cost <- new_cost
                        row_index <- i
                y.[tid] <- row_index

        member this.Apply (x: dMatrix<'T>) =
            let blockSize = 256
            let gridSize = divup x.num_cols blockSize
            let lp = LaunchParam(gridSize, blockSize)
            let dy = this.GPUWorker.Malloc<int>(x.num_cols)
            this.GPULaunch <@ this.Kernel @> lp x.num_cols x.dArray.Ptr dy.Ptr
            dy
