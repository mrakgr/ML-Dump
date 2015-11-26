namespace Mnist

module Bitmap =
    open Types
    open System.Drawing

    let map_slice_to_bitmap (slice : uint8 []) (bitmap : Bitmap) start_x end_x start_y end_y =
        let mutable slice_ind = 0
        for x=start_x to end_x do
            for y=start_y to end_y do
                let c = int slice.[slice_ind]
                slice_ind <- slice_ind+1
                let color = Color.FromArgb(c,c,c)
                bitmap.SetPixel(y,x,color) 

    let make_bitmap_from_imageset (imageset : MnistImageset) x_size y_size =
        let format = System.Drawing.Imaging.PixelFormat.Format24bppRgb
        let bitmap_digit = new Bitmap(imageset.num_cols*x_size,imageset.num_rows*y_size,format)
        let mutable digits = 0
        for x=0 to x_size-1 do
            for y=0 to y_size-1 do
                let start_slice = digits*imageset.num_rows*imageset.num_cols
                let end_slice = (digits+1)*imageset.num_rows*imageset.num_cols-1
                let slice = imageset.raw_data.[start_slice..end_slice]
                digits <- digits+1

                let start_x = x*imageset.num_rows
                let end_x = start_x+imageset.num_rows-1
                let start_y = y*imageset.num_cols
                let end_y = start_y+imageset.num_cols-1
                map_slice_to_bitmap slice bitmap_digit start_x end_x start_y end_y
        bitmap_digit


