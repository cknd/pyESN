# Echo State Networks in Python

[Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network) are easy to train recurrent neural networks. I found the Matlab code in the linked article unnecessarily large and scary, so I started a smaller, self-contained implementation.

It does a bit of input sanitation, accepts various parameters and conforms roughly to the scikit-learn interface. So, in terms of complexity, it sits somewhere between this very helpful [minimal ESN script](http://minds.jacobs-university.de/mantas/code) and something like Jaeger's original [Matlab toolbox](http://www.faculty.jacobs-university.de/hjaeger/pubs/freqGen.zip).

# Examples

- [learning to be a tunable frequency generator](http://nbviewer.ipython.org/github/cknd/pyESN/blob/master/freqgen.ipynb)
- [learning a Mackey-Glass system](http://nbviewer.ipython.org/github/cknd/pyESN/blob/master/mackey.ipynb)