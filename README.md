# QuantumReservoirComputing

[![Build Status](https://travis-ci.com/Z-Denis/QuantumReservoirComputing.jl.svg?branch=master)](https://travis-ci.com/Z-Denis/QuantumReservoirComputing.jl)
[![Codecov](https://codecov.io/gh/Z-Denis/QuantumReservoirComputing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Z-Denis/QuantumReservoirComputing.jl)

Small package to perform not-yet-quantum reservoir computing as in Ref. <sup id="a1">[1](#f1)</sup> in a flexible way. See the example on MNIST bellow.

```julia
using Distributed
# Add N-1 processes in addition to the master one
N = 4
nprocs()<N && addprocs(N-nprocs())
@everywhere using ParallelDataTransfer
@everywhere using QuantumReservoirComputing, LightGraphs, DifferentialEquations, Statistics
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using ProgressMeter
# Set sizes of the training and test sets
M_train = 5000
M_test  = 1000

# Retrieve MNIST dataset
imgs = MNIST.images()
imgs_train = [float.(img) for img in imgs[1:M_train]]
imgs_test  = [float.(img) for img in imgs[M_train+1:M_train+M_test]]
# Send data from master process to all workers
@passobj 1 workers() imgs_train
@passobj 1 workers() imgs_test

# Set the input size of the reservoir to the height of MNIST images
input_size = size(first(imgs),2)
@passobj 1 workers() input_size

# Retrieve labels
labels_train = MNIST.labels()[1:M_train]
labels_test  = MNIST.labels()[M_train+1:M_train+M_test]

# One-hot-encode the labels
Y_train = onehotbatch(labels_train, 0:9)
Y_test  = onehotbatch(labels_test , 0:9)

# Each MNIST image will be encoded into a a set of `input_size` timeseries of
# temporal length tf-ti
@everywhere ti, tf = 0.0, 5.0
# `f1` transforms matrices into timeseries
@everywhere f1(mat) = InputSignal(mat, ti, tf)
# `cgle_layer` defines a reservoir that accepts timeseries of `InputSignal` type
# and outputs the value of the complex fields of a complex Ginzburg-Landau
# equation at time tf.
@everywhere γ, Γ, g = 1e-4, 5., 2. # Reservoir's parameters
@everywhere D = 0.0           # No noise is here considered
@everywhere G = grid([10,10]) # Define the system geometry to be a 2D lattice
@everywhere cgle_layer = GLLayer(ti,tf,input_size,γ,Γ,g,G,D)
# `f2` transforms field values into populations
@everywhere f2(sol) = mean(abs2.(sol), dims=2)[:,1]

println("Computing populations:")
# Compute populations of the training set
pops_train = @time @showprogress "Training set: " pmap(imgs_train) do img
    f2(cgle_layer(f1(img)))
end
X_train = reduce(hcat, pops_train)

# Compute populations of the test set
pops_test = @time @showprogress "Test set: " pmap(imgs_test) do img
    f2(cgle_layer(f1(img)))
end
X_test = reduce(hcat, pops_test)

# Define a simple model classifying inputs from the reservoir populations
m = Chain(Dense(nv(cgle_layer.G), 10),softmax)

# Define the loss function
loss(x,y) = crossentropy(m(x),y)

# Choose an optimizer
opt = ADAM(0.2)

# Iterate over 4000 epochs (this is multithreaded internally by Flux)
println("Learning:")
@time for it=1:4000
    pars = params(m)

    L, back = Tracker.forward(()->loss(X_train,Y_train), pars)
    grads = back(1)

    for p in pars
        Tracker.update!(opt, p, grads[p])
    end

    if mod(it,200)==0
        println(L, "\t,\t", count(onecold(m(X_train)) .== onecold(Y_train)) / length(imgs_train), "\t,\t", count(onecold(m(X_test)) .== onecold(Y_test)) / length(imgs_test))
    end
end
```

Performing the computation of the reservoir's outputs on 4 processes and the optimization on 6 threads, one gets the following (upon second time execution, to avoid timing the compiling time)
```julia-repl
Computing populations:
Training set :100%|███████████████████████████████████| Time: 0:00:39
 39.149486 seconds (1.38 M allocations: 92.152 MiB, 0.08% gc time)
Test set :100%|███████████████████████████████████████| Time: 0:00:07
  7.961126 seconds (417.24 k allocations: 25.769 MiB, 0.11% gc time)
Learning:
1.2234291f0 (tracked)   ,   0.6852  ,   0.654
0.94211835f0 (tracked)  ,   0.7502  ,   0.724
0.80892193f0 (tracked)  ,   0.782   ,   0.768
0.7277918f0 (tracked)   ,   0.801   ,   0.794
0.67145145f0 (tracked)  ,   0.8152  ,   0.809
0.629144f0 (tracked)    ,   0.8284  ,   0.815
0.59567076f0 (tracked)  ,   0.837   ,   0.826
0.56818193f0 (tracked)  ,   0.8444  ,   0.834
0.5449687f0 (tracked)   ,   0.8488  ,   0.84
0.5249333f0 (tracked)   ,   0.8552  ,   0.843
0.50748515f0 (tracked)  ,   0.8604  ,   0.85
0.49216288f0 (tracked)  ,   0.8642  ,   0.853
0.47863045f0 (tracked)  ,   0.8676  ,   0.857
0.4665826f0 (tracked)   ,   0.8702  ,   0.858
0.45567635f0 (tracked)  ,   0.875   ,   0.86
0.44582537f0 (tracked)  ,   0.8794  ,   0.862
0.43687144f0 (tracked)  ,   0.8824  ,   0.864
0.4286812f0 (tracked)   ,   0.8838  ,   0.866
0.42116913f0 (tracked)  ,   0.8846  ,   0.868
0.41418228f0 (tracked)  ,   0.8866  ,   0.869
 37.634682 seconds (2.85 M allocations: 25.643 GiB, 11.53% gc time)
```
i.e. roughly 1'30" of execution time on one tenth of the MNIST dataset.

One can then build the full classifier by function composition as:
```julia
reservoir  = f2 ∘ cgle_layer ∘ f1
classifier = (x->x-1) ∘ onecold ∘ m ∘ reservoir
##

i = rand(1:length(imgs_test)) # = 996
labels_test[i]                # = 7
classifier(imgs_test[i])      # = 7
```

## References

<b id="f1">[1]</b> A. Opala, S. Ghosh, T. C.H. Liew, and M. Matuszewski, Neuromorphic Computing in Ginzburg-Landau Polariton-Lattice Systems, [Phys. Rev. Applied 11, 064029 (2019)](https://doi.org/10.1103/PhysRevApplied.11.064029) [↩](#a1)
