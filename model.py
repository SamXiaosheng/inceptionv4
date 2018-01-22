import tensorflow as tf

from data_generator import MNISTData

images_path = "./MNIST_train_images.gz"
labels_path = "./MNIST_train_labels.gz"
mnist = MNISTData(images_path, labels_path)

num_samples = mnist.inputs.shape[0]
num_rows = mnist.inputs.shape[1]
num_cols = mnist.inputs.shape[2]
num_ch = mnist.inputs.shape[3]
num_class = mnist.outputs.shape[1]

X = tf.placeholder(tf.float32, [None, num_rows, num_cols, num_ch])
y = tf.placeholder(tf.float32, [None, 10])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, num_ch, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

a_conv1 = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(a_conv1 + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

a_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(a_conv2 + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
a_fc1 = tf.matmul(h_pool2_flat, W_fc1)
h_fc1 = tf.nn.relu(a_fc1 + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc1 = tf.Variable(tf.truncated_normal([1024, num_class], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[num_class]))

y_hat = tf.matmul(h_fc1_drop, W_fc1) + b_fc1

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        images, labels = mnist.get_batch(50)
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={X: images, y: labels, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
      
        train_step.run(feed_dict={X: images, y: labels, keep_prob: 0.5})

