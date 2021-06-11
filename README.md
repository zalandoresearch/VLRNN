# Very Long Recurrent Neural Networks
RNN for long sequences usually have an adverse ratio of GPU utilization over memory consumption. Processing a long sequences recurrently in general does not allow parallelization over the time dimension, as future activations depend on past activations. The only option for parallelization is over the batch dimension (increasing the batch size). 
At the same time, long sequences give rise to large memory consumption when computing gradients with common automatic differentiation techniques. Usually, in the forward pass, all activations in all layers and time steps are computed and stored in GPU memory. In the backward pass, the loss is differentiated and deltas are propageted back through the network, where, together with the stored acticvations, they are used to compute the weight updates (https://en.wikipedia.org/wiki/Backpropagation_through_time). The memory demand of stored activations scales linearly with the batch size, hence batch size is no leaver to improve the utilization/memory ratio. The limited GPU memory may disallow reasonable GPU utilization to be achieved. 
## Solution
VLRNN ofer to efficiently compute forward and backward passes of RNN in arbitrarily long sequences. **The memory efficiency comes at the cost of one additional forward pass without gradient computation.**
* works with arbitray RNN architectures that exhibit strictly sequential processing
* multi-layer RNN
* packed sequence (https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html) support for batches of variable length sequences

### Limitations
* overall loss must be linear function of per-timestep losses *l<sub>t</sub>*
* no bi-diretional RNN

## Usage
tbw

## How does it work
The proposed solution in VLRNN is to perform forward/backward computations in blocks of short sequence length such that all activations inside a block fit well into GPU memory for decent batch sizes. In order to compute updates in a block in the middle of the sequence, we need 
* activations *x<sub>t</sub>* at the static input at the block, 
* the latent input *h<sub>t-1</sub>* of the block (the RNN hidden state or memory cells), 
* the delta *𝛿z<sub>t</sub>* flowing into the block from sequence losses, 
* and the deltas *𝛿h<sub>t+𝛥t</sub>* at the end of the block (flowing back from backpropagating the adjacent block). 

![Block RNN Schema](doc/block_rnn.png?raw=true "Title")

Except for the latent (hidden state) activations and deltas, everything is available. For the latent activations we first run a forward pass through the network with gradiend computations disabled and compute (and keep in GPU memory) the activations
![equation](https://latex.codecogs.com/png.latex?%5Cinline%20%5Clarge%20h_%7Bn%5CDelta%20t%7D%2C%5C%20n%3D0%5Cldots%20N-1) for all N blocks.
Then we compute usual forward/backward passed through each block from last to first, collect gardients to all the (shared) weights in the block, and release all activations and deltas of this block, except ![\delta h_t](https://latex.codecogs.com/png.latex?%5Cinline%20%5Clarge%20%5Cdelta%20h_t) at the block input, from GPU memory. ![\delta h_t](https://latex.codecogs.com/png.latex?%5Cinline%20%5Clarge%20%5Cdelta%20h_t) is feed into the backward process of the preceeding block.



## License

The MIT License (MIT)

Copyright (c) 2020 Zalando SE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
