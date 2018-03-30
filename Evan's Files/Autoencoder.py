import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf


# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):

    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

    return mean, stddev


# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y


# Gateway
def autoencoder(x_hat, x, dim_in, dim_z, n_hidden, keep_prob):

    # encoding
    mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # decoding
    y = bernoulli_MLP_decoder(z, n_hidden, dim_in, keep_prob)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
    kl_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    kl_divergence = tf.reduce_mean(kl_divergence)

    ELBO = marginal_likelihood - kl_divergence

    loss = -ELBO

    return y, z, loss, -marginal_likelihood, kl_divergence


def decoder(z, dim_in, n_hidden):

    y = bernoulli_MLP_decoder(z, n_hidden, dim_in, 1.0, reuse=True)

    return y


def build_graph(train_data, n_hidden, n_epochs, batch_size, learn_rate, dim_z):
    """
    This is the main function that generates the encoded variables.
    :param train_data: preprocessed linescans train data and labels (for shuffling), already split
    :param n_hidden: number of hidden variables
    :param n_epochs: number of training iterations
    :param batch_size: size of data for each training iteration
    :param learn_rate: speed at which the encoder will converge
    :param dim_z: arbitrary output vector dimension (latent vector size)
    :return: ???
    """
    dim_in = train_data.shape[1]  # number of dimensions in input data
    train_size = train_data.shape[0]

    # input placeholders
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_in], name='input')
    x = tf.placeholder(tf.float32, shape=[None, dim_in], name='target')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # network architecture
    y, z, loss, neg_marginal_likelihood, kl_divergence = autoencoder(x_hat, x, dim_in, dim_z, n_hidden, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # train
    total_batch = int(train_size / batch_size)
    min_tot_loss = 1e99

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_data)

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (train_size)
                batch_xs_input = train_data[offset:(offset + batch_size), :]

                batch_xs_target = batch_xs_input

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, kl_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob: 0.9})

            # print cost every epoch
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
            epoch, tot_loss, loss_likelihood, loss_divergence))

            # if minimum loss is updated or final epoch, finish
            if min_tot_loss > tot_loss or epoch + 1 == n_epochs:
                print('Done training!')
    return None