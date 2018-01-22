import tensorflow as tf

from data_generator import MNISTData

images_path = "./MNIST_train_images.gz"
labels_path = "./MNIST_train_labels.gz"
mnist = MNISTData(images_path, labels_path)

validate, train = mnist.split(5000)

num_class = train.num_classes

def main():
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, num_class])

    with tf.variable_scope("input"):
        X_p = tf.image.resize_images(X, [299, 299])
    
    with tf.variable_scope("stem"):
        _stem = stem(X_p)
    
    _inception_a = {-1: _stem}
    for i in range(4):
        with tf.variable_scope('inception_a_'+str(i)):
            _inception_a[i] = inception_a(_inception_a[i-1])

    with tf.variable_scope('reduction_a'):
        _reduction_a = reduction_a(_inception_a[3])

    _inception_b = {-1: _reduction_a}
    for i in range(7):
        with tf.variable_scope('inception_b_'+str(i)):
            _inception_b[i] = inception_b(_inception_b[i-1])

    with tf.variable_scope('reduction_b'):
        _reduction_b = reduction_b(_inception_b[6])

    _inception_c = {-1: _reduction_b}
    for i in range(3):
        with tf.variable_scope('inception_c_'+str(i)):
            _inception_c[i] = inception_c(_inception_c[i-1])

    with tf.variable_scope('model'):
        pool = tf.nn.avg_pool(_inception_c[2], [1, 8, 8, 1], [1, 1, 1, 1], padding='VALID', name='pool')
        pool_f = tf.reshape(pool, [-1, 1536])

        W_fc = tf.get_variable('W_fc', [1536, 10])
        b_fc = tf.get_variable('b_fc', [10])

        z_fc = tf.matmul(pool_f, W_fc, name='z_fc')
        h_fc = tf.add(z_fc, b_fc, name='h_fc')

        y_hat = tf.nn.softmax(h_fc, name='y_hat')

        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('train'):
        h_drop = tf.nn.dropout(h_fc, 0.80)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(20):
            images, labels = train.get_batch(1)
            if i % 1 == 0:
                train_accuracy = accuracy.eval(feed_dict={X:images, y: labels})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={X: images, y: labels})


def inception_c(tensor):
    pool_1_1   = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name='pool_1_1')
    conv_1_2   = conv(pool_1_1, 'conv_1_2',   [1, 1, 1536, 256], [1, 1, 1, 1])
    
    conv_2_1   = conv(tensor,   'conv_2_1',   [1, 1, 1536, 256], [1, 1, 1, 1])

    conv_3_1   = conv(tensor,   'conv_3_1',   [1, 1, 1536, 384], [1, 1, 1, 1])
    conv_3_2_1 = conv(conv_3_1, 'conv_3_2_1', [1, 3, 384, 256],  [1, 1, 1, 1])
    conv_3_2_2 = conv(conv_3_1, 'conv_3_2_2', [3, 1, 384, 256],  [1, 1, 1, 1])

    conv_4_1   = conv(tensor,   'conv_4_1',   [1, 1, 1536, 384], [1, 1, 1, 1])
    conv_4_2   = conv(conv_4_1, 'conv_4_2',   [1, 3, 384, 448],  [1, 1, 1, 1])
    conv_4_3   = conv(conv_4_2, 'conv_4_3',   [3, 1, 448, 512],  [1, 1, 1, 1])
    conv_4_3_1 = conv(conv_4_3, 'conv_4_3_1', [1, 3, 512, 256],  [1, 1, 1, 1])
    conv_4_3_2 = conv(conv_4_3, 'conv_4_3_2', [3, 1, 512, 256],  [1, 1, 1, 1])

    concat = tf.concat([conv_1_2, conv_2_1, conv_3_2_1, conv_3_2_2, conv_4_3_1, conv_4_3_2], axis=3, name='concat')

    return concat


def reduction_b(tensor):
    pool_1_1 = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool_1_1')

    conv_2_1 = conv(tensor,   'conv_2_1', [1, 1, 1024, 192], [1, 1, 1, 1])
    conv_2_2 = conv(conv_2_1, 'conv_2_2', [3, 3, 192, 192],  [1, 2, 2, 1], padding='VALID')

    conv_3_1 = conv(tensor,   'conv_3_1', [1, 1, 1024, 256], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [1, 7, 256, 256],  [1, 1, 1, 1])
    conv_3_3 = conv(conv_3_2, 'conv_3_3', [7, 1, 256, 320],  [1, 1, 1, 1])
    conv_3_4 = conv(conv_3_3, 'conv_3_4', [3, 3, 320, 320],  [1, 2, 2, 1], padding='VALID')

    concat = tf.concat([pool_1_1, conv_2_2, conv_3_4], axis=3, name='concat')

    return concat


def inception_b(tensor):
    pool_1_1 = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name='pool_1_1')
    conv_1_2 = conv(pool_1_1, 'conv_1_2', [1, 1, 1024, 128], [1, 1, 1, 1],)

    conv_2_1 = conv(tensor,   'conv_2_1', [1, 1, 1024, 384], [1, 1, 1, 1])

    conv_3_1 = conv(tensor,   'conv_3_1', [1, 1, 1024, 192], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [1, 7, 192, 224],  [1, 1, 1, 1])
    conv_3_3 = conv(conv_3_2, 'conv_3_3', [1, 7, 224, 256],  [1, 1, 1, 1])

    conv_4_1 = conv(tensor,   'conv_4_1', [1, 1, 1024, 192], [1, 1, 1, 1])
    conv_4_2 = conv(conv_4_1, 'conv_4_2', [1, 7, 192, 192],  [1, 1, 1, 1])
    conv_4_3 = conv(conv_4_2, 'conv_4_3', [7, 1, 192, 224],  [1, 1, 1, 1])
    conv_4_4 = conv(conv_4_3, 'conv_4_4', [1, 7, 224, 224],  [1, 1, 1, 1])
    conv_4_5 = conv(conv_4_4, 'conv_4_5', [7, 1, 224, 256],  [1, 1, 1, 1])

    concat = tf.concat([conv_1_2, conv_2_1, conv_3_3, conv_4_5], axis=3, name='concat')

    return concat


def reduction_a(tensor):
    pool_1_1 = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool_1_1')

    conv_2_1 = conv(tensor,   'conv_2_1', [3, 3, 384, 384], [1, 2, 2, 1], padding='VALID')

    conv_3_1 = conv(tensor,   'conv_3_1', [1, 1, 384, 192], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [3, 3, 192, 224], [1, 1, 1, 1])
    conv_3_3 = conv(conv_3_2, 'conv_3_3', [3, 3, 224, 256], [1, 2, 2, 1], padding='VALID')

    concat = tf.concat([pool_1_1, conv_2_1, conv_3_3], axis=3, name='concat')

    return concat


def inception_a(tensor):
    pool_1_1 = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name='pool_1_1')
    conv_1_1 = conv(pool_1_1, 'conv_1_1', [1, 1, 384, 96], [1, 1, 1, 1])

    conv_2_1 = conv(tensor,   'conv_2_1', [1, 1, 384, 96], [1, 1, 1, 1])
    
    conv_3_1 = conv(tensor,   'conv_3_1', [1, 1, 384, 64], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [3, 3, 64, 96],  [1, 1, 1, 1])

    conv_4_1 = conv(tensor,   'conv_4_1', [1, 1, 384, 64], [1, 1, 1, 1])
    conv_4_2 = conv(conv_4_1, 'conv_4_2', [3, 3, 64, 96],  [1, 1, 1, 1])
    conv_4_3 = conv(conv_4_2, 'conv_4_3', [3, 3, 96, 96],  [1, 1, 1, 1])

    concat = tf.concat([conv_1_1, conv_2_1, conv_3_2, conv_4_3], axis=3, name='concat')

    return concat


def stem(tensor):
    conv_1     = conv(tensor,     'conv_1',     [3, 3, 1, 32],   [1, 2, 2, 1], padding='VALID')

    conv_2     = conv(conv_1,     'conv_2',     [3, 3, 32, 32],  [1, 1, 1, 1], padding='VALID')

    conv_3     = conv(conv_2,     'conv_3',     [3, 3, 32, 64],  [1, 1, 1, 1])

    conv_4_1   = conv(conv_3,     'conv_4_1',   [3, 3, 64, 64],  [1, 2, 2, 1], padding='VALID')
    conv_4_2   = conv(conv_3,     'conv_4_2',   [3, 3, 64, 96],  [1, 2, 2, 1], padding='VALID')

    concat_1   = tf.concat([conv_4_1, conv_4_2], axis=3, name='concat_1')

    conv_5_1_1 = conv(concat_1,   'conv_5_1_1', [1, 1, 160, 64], [1, 1, 1, 1])
    conv_5_1_2 = conv(conv_5_1_1, 'conv_5_1_2', [3, 3, 64, 96],  [1, 1, 1, 1], padding='VALID')

    conv_5_2_1 = conv(concat_1,   'conv_5_2_1', [1, 1, 160, 64], [1, 1, 1, 1])
    conv_5_2_2 = conv(conv_5_2_1, 'conv_5_2_2', [7, 1, 64, 64],  [1, 1, 1, 1])
    conv_5_2_3 = conv(conv_5_2_2, 'conv_5_2_3', [1, 7, 64, 64],  [1, 1, 1, 1])
    conv_5_2_4 = conv(conv_5_2_3, 'conv_5_2_4', [3, 3, 64, 96],  [1, 1, 1, 1], padding='VALID')

    concat_2   = tf.concat([conv_5_1_2, conv_5_2_4], axis=3, name='concat_2')

    conv_6_1   = conv(concat_2,   'conv_6_1_1', [3, 3, 192, 192],  [1, 2, 2, 1], padding='VALID')
    pool_6_2   = tf.nn.max_pool(concat_2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool_6_2')

    concat_3   = tf.concat([conv_6_1, pool_6_2], axis=3, name='concat_3')

    return concat_3


def conv(tensor, name, shape, strides=[1, 1, 1, 1], padding='SAME', activation=tf.nn.relu):
    W = tf.get_variable(name+"_W", shape)
    b = tf.get_variable(name+"_b", shape[-1])
    z = tf.nn.conv2d(tensor, W, strides=strides, padding=padding, name=name+'_z')
    h = tf.add(z, b, name=name+'_h')
    a = activation(h, name=name+'_a')

    return a


def fc(tensor, name, dim, activation=tf.nn.relu):
    W = tf.get_variable(name+"_W", [tensor.shape[1], dim])
    b = tf.get_variable(name+"_b", [dim])
    z = tf.matmul(tensor, W, name=name+"_z")
    h = tf.add(z, b, name=name+'_h')
    a = activation(h, name=name+'_a')

    return a


if __name__ == '__main__':
    main()

