# Music-Generator
This is a basic project that implements Restricted Boltzmann Machines(RBM) to generate music.

### Gibb's Sampling
A good idea would be for the model to try to reconstruct the original music, but not exactly the same. The model tries to reconstruct the input data through an RBM which is made from scratch using [Tensorflow](https://www.tensorflow.org).

The values of the hidden nodes conditioned on observing the value of the visible layer i.e. p(h|x) are first sampled. Since each node is conditionally independent, we can carry out [Bernoulli Sampling](http://www.asasrms.org/Proceedings/y2002/Files/JSM2002-001080.pdf) i.e. if the probability of hidden node being 1 given the visible node is greater than a random value sampled from a uniform distribution between 0 and 1, then the hidden node can be assigned the value 1, else 0.

Once this is done, the visible layer is reconstructed by sampling from p(x|h)

Since, the objective is not to completely mimic the input data but learn from it and generate new music, Gibb's sampling is done for only one iteration.

An elaborated discussion on RBMs can be found [here](https://iq.opengenus.org/restricted-boltzmann-machine/).
