namespace Mnist

module Types =
    open MathNet.Numerics.LinearAlgebra

    let MnistPath = @"C:\!NN\"
    let testSetData = "t10k-images.idx3-ubyte"
    let testSetLabels = "t10k-labels.idx1-ubyte"
    let trainSetData = "train-images.idx3-ubyte"
    let trainSetLabels = "train-labels.idx1-ubyte"

    open System
    open System.IO

    type MnistImageset = {
        num_images : int32
        num_rows : int32
        num_cols : int32
        raw_data : uint8 []
        raw_labels : uint8 []
        matrix : Matrix<float32>
        label_matrix : Matrix<float32>
        }

    let readInt32 (r : BinaryReader) =
        let arr = r.ReadBytes(4)
        arr |> Array.Reverse
        BitConverter.ToInt32(arr,0)

    let make_imageset data_path label_path =
        use stream_data = File.OpenRead(MnistPath+data_path)
        use stream_label = File.OpenRead(MnistPath+label_path)
        use reader_data = new BinaryReader(stream_data)
        use reader_label = new BinaryReader(stream_label)

        let magic_number = readInt32 reader_data
        let num_images = readInt32 reader_data
        let num_rows = readInt32 reader_data
        let num_cols = readInt32 reader_data
        let total_num_bytes = num_images * num_rows * num_cols

        let raw_data = reader_data.ReadBytes(total_num_bytes)
        let raw_label_data = reader_label.ReadBytes(num_images+8)

        let float_pixel_values = [|for x in raw_data -> (float32 x)/255.0f|]

        let label_matrix = DenseMatrix.zero 10 num_images
        let mutable c = 0
        for x in raw_label_data.[8..] do
            label_matrix.[int x, c] <- 1.0f
            c <- c+1
        {
        num_images = num_images
        num_rows = num_rows
        num_cols = num_cols
        raw_data = raw_data
        raw_labels = raw_label_data.[8..]
        matrix = DenseMatrix.raw (num_rows*num_cols) num_images float_pixel_values // Creates the matrix.
        label_matrix = label_matrix
        }

