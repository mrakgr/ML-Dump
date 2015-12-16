The stuff I've written over the last four months for machine learning. Some of the early examples might be broken and I can't yet get the GRU and the LSTM to work, but the stuff in the 'utils.fsx' file works nicely and has everything one might need to write a ML algorithm. 'convolution.fsx' has the things one would need to make a convolutional net. Most examples require the Alea CUDA package and the CUDA 6.5 and the cuDNN 2v library.

Given the great difficulty of writting the more complex algorithms, right now I am studying <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">Automatic Differentiation</a>.

I plan to write a bunch of tutorials based on the above after new years' which should put this dump in order. <del>Hopefully I will be able to complete the LSTM using the reverse mode AD that I've made myself by then.</del>

Done. In the AD directory, ad_utils_v3.fsx has everything one might need to use the basic reverse mode AD on the GPU and the ad_lstm_v3.fsx has an example of use. The API is Spartan, but I've verified what it works. What I failed to do for the entire month of November I did in a week by moving up a level in abstraction.

In the past I've been enthused about neuromorphic chips, but now I see what a difference good software tools can make. With feedforward nets it is not so bad though it took me some practice to make them, but anything more complex than a simple recurrent net is pretty much impossible to get right by hand.

The <a href="http://diffsharp.github.io/DiffSharp/">DiffSharp library</a> has more advanced functionality (though no GPU support at the moment.) I might look into putting the stuff that I did here into the library's backend which should make it much more useful than just leaving it lying around here. While F# is a wonderful language, it – and Windows tools in general for machine learning - are sorely lacking at the moment. The DL community at large is very Python and Linux centric and a lot of DL libraries are difficult to even install, let alone use.

The majority of ML libraries that exist right now are like mushrooms springing up after a rain and will wither away in a few days, but I think the AD focused ones (like Tensorflow and DiffSharp, but not Theano) have the potential for lasting use.

Well, since I got this much use from moving up a level in abstraction, I am currently interested in learning more AD. After that I will write a bunch of tutorials on ML and get this dump in order before moving on, maybe onto the DiffSharp backend.

UPDATE(12/16/2015): I rewrote the ad_utils_v3 using ManagedCuda instead of Alea and amazingly, it is 4.5 times faster. The first version of the new basic reverse AD library can be found in the Spiral Demo v0 folder along with comparisons using DiffSharp and my old library using Alea.

The embedded Reber grammar that I trained the LSTM on is basically just a long sequence of small matrix multiplies (like 64x310), so it might be even relatively faster compared to the CPU version using larger matrices. I am not sure whether Alea is slow or whether the move to cuBLAS 7.5 is the reason for the dramatic speedup, but I like it. This was definitely a lucky windfall for me as without deciding to do the backend in Cuda for DiffSharp I would never have figured out how much I was missing out on.

The basic reverse AD library Spiral is very much usable now despite its infancy and I hope that in the future I can implement parts of it into the DiffSharp library to serve as its backend.

The demo directory is likely to grow stale with time, so I'll dedicate a Github repository to it and post a link here.