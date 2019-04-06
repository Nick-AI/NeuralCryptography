import os
import numpy as np
import keras.backend as K
from tqdm import tqdm as tqdm
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv1D, Reshape, Flatten


class CryptoGAN:

    def __init__(self, n_bits, batch_size=4096, model_dir='./models/', name_add=''):
        np.random.seed(13)
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.name_add = name_add

        # HPs dictated by paper
        self.n_bits = n_bits
        self.nb_epochs = 150000
        self.act = 'sigmoid'
        self.lr = 0.0008
        self.opt = Adam(lr=self.lr)
        self.batch_size = batch_size

        # adversary network
        self.eve = self._get_network(self.n_bits)
        self.eve.compile(loss=['mae'], optimizer=self.opt)

        # a&b networks and full GAN
        self.ali = self._get_network(2*self.n_bits)
        self.bob = self._get_network(2*self.n_bits)

        inp_a = Input(shape=(2*self.n_bits,))
        inp_b = Input(shape=(2*self.n_bits,))
        inp_e = Input(shape=(self.n_bits,))

        reconst_b = self.bob(inp_b)
        reconst_e = self.eve(inp_e)

        # Fix adversary's weights for full GAN
        self.eve.trainable = False

        self.full_model = Model(inputs=[inp_a, inp_b, inp_e],
                                outputs=[reconst_b, reconst_e])
        self.full_model.compile(optimizer=self.opt,
                                loss=['mae', self._adversarial_loss_func])

        # self.eve.summary()
        # self.ali.summary()
        # self.bob.summary()


    # function defining network architecture
    def _get_network(self, input_dims):
        inp = Input(shape=(input_dims,))

        _ = Dense(units=2*self.n_bits, activation=self.act)(inp)
        _ = Reshape(target_shape=(2*self.n_bits, 1))(_)
        _ = Conv1D(kernel_size=4, strides=1, filters=2, activation=self.act, padding='same')(_)
        _ = Conv1D(kernel_size=2, strides=2, filters=2, activation=self.act, padding='same')(_)
        _ = Conv1D(kernel_size=1, strides=1, filters=1, activation=self.act, padding='same')(_)
        _ = Conv1D(kernel_size=1, strides=1, filters=1, activation='tanh', padding='same')(_)
        out = Flatten()(_)

        return Model(inputs=[inp], outputs=out)

    # Custom loss function based on definition in paper
    def _adversarial_loss_func(self, y_true, y_pred):
        rand_guess = K.variable(np.array([self.n_bits/2]))
        return K.square(rand_guess - K.mean(K.abs(y_true - y_pred))) / K.square(rand_guess)

    # function to query for key/ plaintext values
    def _gen_bit_string(self):
        seq = np.random.randint(0, 2, self.n_bits)
        return seq

    # generate train batches
    def _gen_train_data(self, multi_batch=1):
        pt_batch = []
        key_batch = []
        for _ in range(multi_batch * self.batch_size):
            pt_batch.append(self._gen_bit_string())
            key_batch.append(self._gen_bit_string())
        pt_batch = np.array(pt_batch)
        key_batch = np.array(key_batch)
        ct_batch = np.round(self.ali.predict(np.concatenate((pt_batch, key_batch), axis=1)))  # making it binary
        train_batch = np.array([pt_batch, key_batch, ct_batch])
        return train_batch

    def _stop_check(self):
        # Stopping criteria are implemented as defined in the paper
        end_flag = 0

        batch = self._gen_train_data()
        pt_batch = batch[0, :, :]
        key_batch = batch[1, :, :]
        ct_batch = batch[2, :, :]
        inp_a = np.concatenate((pt_batch, key_batch), axis=1)
        inp_b = np.concatenate((ct_batch, key_batch), axis=1)

        preds = self.full_model.predict_on_batch(x=[inp_a, inp_b, ct_batch])

        # predictions are rounded to bits for evaluation purposes
        bob_reconst = np.round(preds[0])
        eve_reconst = np.round(preds[1])

        # reconstruction criterium between alice and bob
        if np.abs(np.mean(bob_reconst - pt_batch)) < 0.05:
            end_flag += 0.3

        # reconstruction criterium between eve and a&b
        correct = (eve_reconst == pt_batch)
        avg_correct = np.mean(np.sum(correct, axis=1))

        if self.n_bits/2-2 <= avg_correct <= self.n_bits/2+2:
            end_flag += 0.3

        return np.round(end_flag)

    def train_full(self):
        ab_loss = []
        e_loss = []
        ep_batches = 3  # generate all batches needed for epoch at once, speeds up training
        assert ep_batches%3 == 0

        for ep in tqdm(range(int(3*self.nb_epochs/ep_batches))):
            # train encryption nets
            trn_batch = self._gen_train_data(ep_batches)
            for bidx in range(0, int(ep_batches/3)):
                batch = trn_batch[:, bidx:(bidx+1)*self.batch_size, :]
                pt_batch = batch[0, :, :]
                key_batch = batch[1, :, :]
                ct_batch = batch[2, :, :]
                inp_a = np.concatenate((pt_batch, key_batch), axis=1)
                inp_b = np.concatenate((ct_batch, key_batch), axis=1)

                # ['loss' (overall), 'model_3_loss' (bob), 'model_1_loss' (eve)]
                ab_loss.append(self.full_model.train_on_batch(x=[inp_a, inp_b, ct_batch],
                                                              y=[pt_batch, pt_batch]))

            # train attacker net (twice as many iterations as encryption nets)
            for bidx in range(int(ep_batches/3), ep_batches):
                batch = trn_batch[:, (bidx)*self.batch_size:(bidx+1)*self.batch_size, :]
                pt_batch = batch[0, :, :]
                ct_batch = batch[2, :, :]
                e_loss.append(self.eve.train_on_batch(x=ct_batch,
                                                      y=pt_batch))

            if ep % 100 == 0:  # for efficiency reasons, stopping criterium is only checked in intervals
                if self._stop_check():
                    import pdb
                    pdb.set_trace()
                    self.ali.save_weights(self.model_dir + f'{self.name_add}ali_{ep}epochs_succs.mdl')
                    self.bob.save_weights(self.model_dir + f'{self.name_add}bob_{ep}epochs_succs.mdl')
                    self.eve.save_weights(self.model_dir + f'{self.name_add}eve_{ep}epochs_succs.mdl')

        self.ali.save_weights(self.model_dir + f'{self.name_add}ali_{ep}epochs_faild.mdl')
        self.bob.save_weights(self.model_dir + f'{self.name_add}bob_{ep}epochs_faild.mdl')
        self.eve.save_weights(self.model_dir + f'{self.name_add}eve_{ep}epochs_faild.mdl')

        return ab_loss, e_loss


if __name__ == '__main__':
    for i in range(10):
        model = CryptoGAN(32, name_add=f'{i}_')
        model.train_full()


