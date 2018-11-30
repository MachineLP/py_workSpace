# coding=utf-8

import tensorflow as tf 

def gan_log_loss(pos, neg, name='gan_log_loss'):
    """
    log loss function for GANs.
    - Generative Adversarial Networks: https://arxiv.org/abs/1406.2661
    """
    with tf.variable_scope(name):
        # generative model G
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.ones_like(neg)))
        # discriminative model D
        d_loss_pos = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos, labels=tf.ones_like(pos)))
        d_loss_neg = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.zeros_like(neg)))
        d_loss = tf.add(.5 * d_loss_pos, .5 * d_loss_neg)
    return g_loss, d_loss

def gan_ls_loss(pos, neg, value=1., name='gan_ls_loss'):
    """
    gan with least-square loss
    """
    with tf.variable_scope(name):
        l2_pos = tf.reduce_mean(tf.squared_difference(pos, value))
        l2_neg = tf.reduce_mean(tf.square(neg))
        d_loss = tf.add(.5 * l2_pos, .5 * l2_neg)
        g_loss = tf.reduce_mean(tf.squared_difference(neg, value))
    return g_loss, d_loss

def gan_wgan_loss(pos, neg, name='gan_wgan_loss'):
    """
    wgan loss function for GANs.
    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(neg-pos)
        g_loss = -tf.reduce_mean(neg)
    return g_loss, d_loss

def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs
    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))


def discriminator_loss(loss_func, real, fake, real_blur):
    real_loss = 0
    fake_loss = 0
    real_blur_loss = 0


    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)
        real_blur_loss = tf.reduce_mean(real_blur)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.square(real - 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))
        real_blur_loss = tf.reduce_mean(tf.square(real_blur))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
        real_blur_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
        real_blur_loss = tf.reduce_mean(relu(1.0 + real_blur))

    loss = real_loss + fake_loss + real_blur_loss

    return loss

def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss



if __name__ == "__main__":

    LAMBDA = 10
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator_dcgan')
    D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Discriminator_dcgan')

    disc_fake = discriminator(fake_data)
    disc_real = discriminator(real_data)
    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=G_vars)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=D_vars)

        clip_ops = []
        for var in disc_params:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var, 
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Gradient penalty
        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=G_vars)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=D_vars)

    elif MODE == 'dcgan':
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
        disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
        disc_cost /= 2.

        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=G_vars)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost, var_list=D_vars)


