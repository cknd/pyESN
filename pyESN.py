import numpy as np

class ESN():

    def _correct_dimensions(self,s,targetlength):
        """checks the dimensionality of some numeric argument s. s can be None,
        a scalar or a 1D array of correct length. If it's a scalar, it gets
        broadcasted to the necessary length."""
        if s is not None:
            s = np.array(s)
            #import ipdb; ipdb.set_trace()
            if s.ndim == 0:
                s = np.array([s]*targetlength)
            elif s.ndim == 1:
                if not len(s) == targetlength:
                    raise ValueError("Vector needs to be of length"+str(targetlength))
            else:
                raise ValueError("Invalid argument")
        return s


    def __init__(self,n_inputunits,n_outunits,n_reservoir=99,spectral_radius=0.95,sparsity=0,noise=0.001,
                 input_shift=None,input_scaling=None,feedback_scaling=None,teacher_scaling=None,teacher_shift=None,
                 output_activation=lambda x:x,inverse_output_activation=lambda x:x,random_state=None,teacher_forcing=True,silent=True):
        self.n_inputunits = self._correct_dimensions(n_inputunits,1)
        self.n_reservoir = self._correct_dimensions(n_reservoir,1)
        self.n_outunits = self._correct_dimensions(n_outunits,1)
        self.spectral_radius = self._correct_dimensions(spectral_radius,1)
        self.sparsity = self._correct_dimensions(sparsity,1)
        self.noise = self._correct_dimensions(noise,1)

        self.input_shift = self._correct_dimensions(input_shift,n_inputunits)
        self.input_scaling = self._correct_dimensions(input_scaling,n_inputunits)

        self.feedback_scaling = self._correct_dimensions(feedback_scaling,1)
        self.teacher_scaling = self._correct_dimensions(teacher_scaling,1)
        self.teacher_shift = self._correct_dimensions(teacher_shift,1)

        self.output_activation = output_activation
        self.inverse_output_activation = inverse_output_activation
        self.random_state = random_state

        # the given random_state might be either an actual RandomState object, a seed or nothing
        # (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: "+str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.initweights()

    def initweights(self):
        # initialize recurrent weights:
        W = self.random_state_.rand(self.n_reservoir,self.n_reservoir) - 0.5
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0 # setting (self.sparsity) percent of connections to zero
        radius = np.max(np.abs(np.linalg.eigvals(W))) # spectral radius of the recurrent weights
        self.W = W*(self.spectral_radius/radius) # rescale them to reach the requested spectral radius

        # initialize input, output, and feedback weights:
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputunits)*2-1
        self.W_out = np.zeros((self.n_outunits, self.n_reservoir + self.n_inputunits))
        self.W_feedb = (self.random_state_.rand(self.n_reservoir, self.n_outunits)*2-1)*(self.feedback_scaling if self.feedback_scaling else 1)

    def _update(self,state,input_pattern,output_pattern):
        if self.teacher_forcing:
            preactivation = np.dot(self.W,state) + np.dot(self.W_in,input_pattern) + np.dot(self.W_feedb,output_pattern)
        else:
            preactivation = np.dot(self.W,state) + np.dot(self.W_in,input_pattern)
        return np.tanh(preactivation) + self.noise*(self.random_state_.rand(self.n_reservoir)-0.5)

    def _scale_inputs(self,inputs):
        if self.input_scaling is not None:
            inputs = np.dot(inputs,np.diag(self.input_scaling)) # multiply j'th input feature by j'th entry of input_scaling
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self,teacher):
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self,teacher_scaled):
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self,inputs,outputs,inspect=False):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs,(len(inputs),-1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs,(len(outputs),-1))
        # adjust input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent: print("harvesting states...")
        # ...by iterating through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1,inputs.shape[0]):
            states[n,:] = self._update(states[n-1],inputs_scaled[n,:],teachers_scaled[n-1,:])

        # learn weights:
        if not self.silent: print("fitting...")
        transient = min(int(inputs.shape[1]/10),100)
        extended_states = np.hstack((states,inputs_scaled))
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:,:]),self.inverse_output_activation(teachers_scaled[transient:,:])).T

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

        if not self.silent: print("training error:")
        pred_train = self._unscale_teacher(self.output_activation(np.dot(extended_states,self.W_out.T)))
        if not self.silent: print(np.sqrt(np.mean((pred_train - outputs)**2)))
        return pred_train


    def predict(self,inputs,continuation=True):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs,(len(inputs),-1))
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
            states[n,:] = self._update(states[n-1,:],inputs[n,:],outputs[n-1,:])
            outputs[n,:] = self.output_activation(np.dot(self.W_out, np.concatenate([states[n,:],inputs[n,:]]) ))

        return self._unscale_teacher(self.output_activation(outputs[1:]))
