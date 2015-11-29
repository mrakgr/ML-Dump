The stuff I've written over the last four months for machine learning. Some of the early examples might be broken and I can't yet get the GRU and the LSTM to work, but the stuff in the 'utils.fsx' file works nicely and has everything one might need to write a ML algorithm. 'convolution.fsx' has the things one would need to make a convolutional net. Most examples require the Alea CUDA package and the <del>CUDA 6.5</del> CUDA 7.5 SDK and the cuDNN 2v library.

Given the great difficulty of writting the more complex algorithms, right now I am studying <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">Automatic Differentiation</a>.

I plan to write a bunch of tutorials based on the above after new years' which should put this dump in order. Hopefully I will be able to complete the LSTM using the reverse mode AD that I've made myself by then.
