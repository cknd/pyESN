# Using an Echo State Network to learn a dynamical system

This is a simplified implementation of Herbert Jaeger's Echo State
Network [learning to be a frequency generator](http://www.scholarpedia.org/article/Echo_state_network)
controlled by an external signal. Echo State Networks are easy to train recurrent neural networks.

This is not a tutorial on ESNs, but there are some colorful plots at the end.

It's an Ipython notebook file - [view it on nbviewer](http://nbviewer.ipython.org/github/cluclu/pyESN/blob/master/ESN_freqgen.ipynb).

The ESN class here is equivalent to some subset of Jaeger's [original Matlab code](http://www.faculty.jacobs-university.de/hjaeger/pubs/freqGen.zip) &mdash;
I wanted a simpler version because that code is capable of many wonders and consists of three
dozen Matlab files, which could perhaps scare someone.

Note: This here works (it seems), but don't reuse the code for anything important -- I just left it here when someone asked for a self-contained
example of an ESN doing something interesting. I *might* document and test the ESN class at some later point.
