import numpy as np

class ESN():
    def __init__(self,n_inputunits,n_reservoir,n_outunits,spectral_radius,sparsity=0.2,noise=0.001,
                 input_shift=None,input_scaling=None,feedback_scaling=None,teacher_scaling=None,teacher_shift=None,
                 output_activation=lambda x:x,inverse_output_activation=lambda x:x,random_state=None):
        self.n_inputunits = n_inputunits
        self.n_reservoir = n_reservoir
        self.n_outunits = n_outunits
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = input_shift
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.output_activation = output_activation
        self.inverse_output_activation = inverse_output_activation
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object, a seed or nothing
        # (in which case we use the builtin one)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: "+str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        W = np.random.rand(self.n_reservoir,self.n_reservoir) - 0.5
        W[np.random.rand(*W.shape) < self.sparsity] = 0 # setting (self.sparsity) percent of connections to zero
        radius = np.max(np.abs(np.linalg.eigvals(W))) # find the spectral radius the recurrent weights
        self.W = W*(self.spectral_radius/radius) # rescale them to reach the requested spectral radius

        # initialize input, output, and feedback weights:
        self.W_in = np.random.rand(self.n_reservoir, self.n_inputunits)*2-1
        self.W_out = np.zeros((self.n_outunits, self.n_reservoir + self.n_inputunits))
        self.W_feedb = (np.random.rand(self.n_reservoir, self.n_outunits)*2-1)*self.feedback_scaling

    def _update(self,state,input_pattern,output_pattern):
        preactivation = np.dot(self.W,state) + np.dot(self.W_in,input_pattern) + np.dot(self.W_feedb,output_pattern)
        return np.tanh(preactivation) + self.noise*(np.random.rand(self.n_reservoir)-0.5)

    def _scale_inputs(self,inputs):
        if self.input_scaling:
            inputs = np.dot(inputs,np.diag(self.input_scaling)) # multiply j'th column by j'th entry of input_scaling
        if self.input_shift:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self,teacher):
        if self.teacher_scaling:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self,teacher_scaled):
        if self.teacher_shift:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self,inputs,outputs,inspect=False):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs,(-1,len(inputs)))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs,(-1,len(outputs)))
        # adjust input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        print "harvesting states..."
        # ...by iterating through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1,inputs.shape[0]):
            states[n,:] = self._update(states[n-1],inputs_scaled[n],teachers_scaled[n-1])

        # learn weights:
        print "fitting..."
        extended_states = np.hstack((states,inputs_scaled))
        self.W_out = np.dot(np.linalg.pinv(extended_states[100:,:]),self.inverse_output_activation(teachers_scaled[100:,:])).T

        # remember the last state
        self.laststate = states[-1,:]
        self.lastinput = inputs[-1,:]
        self.lastoutput = outputs[-1,:]

        # visualize the state:
        if inspect:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(states.shape[0]*0.0025,states.shape[1]*0.01))
            plt.imshow(extended_states.T,aspect='auto',interpolation='nearest')
            plt.colorbar()

        print "training error:"
        pred_train = self.predict(inputs,continuation=False)
        print np.sqrt(np.mean((pred_train - outputs)**2))
        return pred_train


    def predict(self,inputs,continuation=True):
        n_samples = inputs.shape[0]
        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputunits)
            lastoutput = np.zeros(self.n_outunits)

        inputs  = np.vstack([lastinput, self._scale_inputs(inputs)])
        states  = np.vstack([laststate, np.zeros((n_samples,  self.n_reservoir))])
        outputs = np.vstack([lastoutput,np.zeros((n_samples,   self.n_outunits))])


        for n in range(1,n_samples):
            states[n,:] = self._update(states[n-1],inputs[n],outputs[n-1])
            outputs[n,:] = self.output_activation(np.dot(self.W_out, np.concatenate([states[n,:],inputs[n,:]]) ))

        return self._unscale_teacher(self.output_activation(outputs[1:]))
