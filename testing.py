import unittest
import numpy as np

from pyESN import ESN

N_in, N, N_out = 5, 75, 3


def random_task():
    X = np.random.randn(100, N_in)
    y = np.random.randn(100, N_out)
    Xp = np.random.randn(50, N_in)
    return X, y, Xp


class RandomStateHandling(unittest.TestCase):

    def setUp(self):
        self.task = random_task()

    def _compare(self, esnA, esnB, should_be):
        """helper function to see if two esns are the same"""
        X, y, Xp = self.task
        test = self.assertTrue if should_be == "same" else self.assertFalse
        test(np.all(np.equal(esnA.W, esnB.W)))
        test(np.all(np.equal(esnA.W_in, esnB.W_in)))
        test(np.all(np.equal(esnA.W_feedb, esnB.W_feedb)))
        test(np.all(np.equal(esnA.fit(X, y), esnB.fit(X, y))))
        test(np.all(np.equal(esnA.W_out, esnB.W_out)))
        test(np.all(np.equal(esnA.predict(Xp), esnB.predict(Xp))))

    def test_integer(self):
        """two esns with the same seed should be the same"""
        esnA = ESN(N_in, N_out, random_state=1)
        esnB = ESN(N_in, N_out, random_state=1)
        self._compare(esnA, esnB, should_be="same")

    def test_randomstate_object(self):
        """two esns with the same randomstate objects should be the same"""
        rstA = np.random.RandomState(1)
        esnA = ESN(N_in, N_out, random_state=rstA)
        rstB = np.random.RandomState(1)
        esnB = ESN(N_in, N_out, random_state=rstB)
        self._compare(esnA, esnB, should_be="same")

    def test_none(self):
        """two esns with no specified seed should be different"""
        esnA = ESN(N_in, N_out, random_state=None)
        esnB = ESN(N_in, N_out, random_state=None)
        self._compare(esnA, esnB, should_be="different")

    def test_nonsense(self):
        """parameter random_state should only accept positive integers"""
        with self.assertRaises(ValueError):
            ESN(N_in, N_out, random_state=-1)

        with self.assertRaises(Exception) as cm:
            ESN(N_in, N_out, random_state=0.5)
        self.assertIn("Invalid seed", str(cm.exception))

    def test_serialisation(self):
        import pickle
        import io
        esn = ESN(N_in, N_out, random_state=1)
        with io.BytesIO() as buf:
            pickle.dump(esn, buf)
            buf.flush()
            buf.seek(0)
            esn_unpickled = pickle.load(buf)
        self._compare(esn, esn_unpickled, should_be='same')


class InitArguments(unittest.TestCase):

    def setUp(self):
        self.X, self.y, self.Xp = random_task()

    def test_inputscaling(self):
        """input scaling factors of different formats should be correctly intereted or rejected"""
        esn = ESN(N_in, N_out, input_scaling=2)
        self.assertTrue(np.all(2 * self.X == esn._scale_inputs(self.X)))
        esn.fit(self.X, self.y)
        esn.predict(self.Xp)

        esn = ESN(N_in, N_out, input_scaling=[2] * N_in)
        self.assertTrue(np.all(2 * self.X == esn._scale_inputs(self.X)))
        esn.fit(self.X, self.y)
        esn.predict(self.Xp)

        esn = ESN(N_in, N_out, input_scaling=np.array([2] * N_in))
        self.assertTrue(np.all(2 * self.X == esn._scale_inputs(self.X)))
        esn.fit(self.X, self.y)
        esn.predict(self.Xp)

        with self.assertRaises(ValueError):
            esn = ESN(N_in, N_out, input_scaling=[2] * (N_in + 1))

        with self.assertRaises(ValueError):
            esn = ESN(N_in, N_out, input_scaling=np.array([[2] * N_in]))

    def test_inputshift(self):
        """input shift factors of different formats should be correctly interpreted or rejected"""
        esn = ESN(N_in, N_out, input_shift=1)
        self.assertTrue(np.all(1 + self.X == esn._scale_inputs(self.X)))
        esn.fit(self.X, self.y)
        esn.predict(self.Xp)

        esn = ESN(N_in, N_out, input_shift=[1] * N_in)
        self.assertTrue(np.all(1 + self.X == esn._scale_inputs(self.X)))
        esn.fit(self.X, self.y)
        esn.predict(self.Xp)

        esn = ESN(N_in, N_out, input_shift=np.array([1] * N_in))
        self.assertTrue(np.all(1 + self.X == esn._scale_inputs(self.X)))
        esn.fit(self.X, self.y)
        esn.predict(self.Xp)

        with self.assertRaises(ValueError):
            esn = ESN(N_in, N_out, input_shift=[1] * (N_in + 1))

        with self.assertRaises(ValueError):
            esn = ESN(N_in, N_out, input_shift=np.array([[1] * N_in]))

    def test_IODimensions(self):
        """try different combinations of input & output dimensionalities & teacher forcing"""
        tasks = [(1, 1, 100, True), (10, 1, 100, True), (1, 10, 100, True), (10, 10, 100, True),
                 (1, 1, 100, False), (10, 1, 100, False), (1, 10, 100, False), (10, 10, 100, False)]
        for t in tasks:
            N_in, N_out, N_samples, tf = t
            X = np.random.randn(
                N_samples, N_in) if N_in > 1 else np.random.randn(N_samples)
            y = np.random.randn(
                N_samples, N_out) if N_out > 1 else np.random.randn(N_samples)
            Xp = np.random.randn(
                N_samples, N_in) if N_in > 1 else np.random.randn(N_samples)
            esn = ESN(N_in, N_out, teacher_forcing=tf)
            prediction_tr = esn.fit(X, y)
            prediction_t = esn.predict(Xp)
            self.assertEqual(prediction_tr.shape, (N_samples, N_out))
            self.assertEqual(prediction_t.shape, (N_samples, N_out))


class Performance(unittest.TestCase):
    # Slighty bending the concept of a unit test, I want to catch performance changes during refactoring.
    # Ideally, this will expand to a collection of known tasks.

    def test_mackey(self):
        try:
            data = np.load('mackey_glass_t17.npy')
        except IOError:
            self.skipTest("missing data")

        esn = ESN(n_inputs=1,
                  n_outputs=1,
                  n_reservoir=500,
                  spectral_radius=1.5,
                  random_state=42)

        trainlen = 2000
        future = 2000
        esn.fit(np.ones(trainlen), data[:trainlen])
        prediction = esn.predict(np.ones(future))
        error = np.sqrt(
            np.mean((prediction.flatten() - data[trainlen:trainlen + future])**2))
        self.assertAlmostEqual(error, 0.1396039098653574)

    def test_freqgen(self):
        rng = np.random.RandomState(42)

        def frequency_generator(N, min_period, max_period, n_changepoints):
            """returns a random step function + a sine wave signal that
               changes its frequency at each such step."""
            # vector of random indices < N, padded with 0 and N at the ends:
            changepoints = np.insert(np.sort(rng.randint(0, N, n_changepoints)), [
                                     0, n_changepoints], [0, N])
            # list of interval boundaries between which the control sequence
            # should be constant:
            const_intervals = list(
                zip(changepoints, np.roll(changepoints, -1)))[:-1]
            # populate a control sequence
            frequency_control = np.zeros((N, 1))
            for (t0, t1) in const_intervals:
                frequency_control[t0:t1] = rng.rand()
            periods = frequency_control * \
                (max_period - min_period) + max_period

            # run time through a sine, while changing the period length
            frequency_output = np.zeros((N, 1))
            z = 0
            for i in range(N):
                z = z + 2 * np.pi / periods[i]
                frequency_output[i] = (np.sin(z) + 1) / 2
            return np.hstack([np.ones((N, 1)), 1 - frequency_control]), frequency_output

        N = 15000
        min_period = 2
        max_period = 10
        n_changepoints = int(N / 200)
        frequency_control, frequency_output = frequency_generator(
            N, min_period, max_period, n_changepoints)

        traintest_cutoff = int(np.ceil(0.7 * N))
        train_ctrl, train_output = frequency_control[
            :traintest_cutoff], frequency_output[:traintest_cutoff]
        test_ctrl, test_output = frequency_control[
            traintest_cutoff:], frequency_output[traintest_cutoff:]

        esn = ESN(n_inputs=2,
                  n_outputs=1,
                  n_reservoir=200,
                  spectral_radius=0.25,
                  sparsity=0.95,
                  noise=0.001,
                  input_shift=[0, 0],
                  input_scaling=[0.01, 3],
                  teacher_scaling=1.12,
                  teacher_shift=-0.7,
                  out_activation=np.tanh,
                  inverse_out_activation=np.arctanh,
                  random_state=rng,
                  silent=True)

        pred_train = esn.fit(train_ctrl, train_output)
        # print "test error:"
        pred_test = esn.predict(test_ctrl)
        error = np.sqrt(np.mean((pred_test - test_output)**2))
        self.assertAlmostEqual(error, 0.30519018985725715)


if __name__ == '__main__':
    unittest.main()
