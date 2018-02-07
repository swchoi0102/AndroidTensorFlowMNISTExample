from __future__ import print_function
import shutil
import os.path
from layers import *
from tensorflow.examples.tutorials.mnist import input_data

EXPORT_DIR = './model'

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create Model
def conv_net(x, weights1, weights2, weights3, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = separable_conv2d(x, weights1['wc1'], weights2['wc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = separable_conv2d(conv1, weights1['wc2'], weights2['wc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.layers.flatten(conv2)
    fc1 = tf.add(tf.matmul(fc1, pruning.apply_mask(weights3['wd1'])), weights3['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights3['wout']), weights3['bout'])
    return out


# Store layers weight & bias
# weight of depthwise filters
weights1 = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 2])),
}

# weight of pointwise filters
weights2 = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 1, 32, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([1, 1, 64, 64])),
}

# weights and bias of fully connected layers
weights3 = {
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'wout': tf.Variable(tf.random_normal([1024, n_classes])),
    # bias
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bout': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = conv_net(x, weights1, weights2, weights3, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
    print("Testing Accuracy: {}".format(test_acc))

    WDC1 = weights1['wc1'].eval(sess)
    WPC1 = weights2['wc1'].eval(sess)
    WDC2 = weights1['wc2'].eval(sess)
    WPC2 = weights2['wc2'].eval(sess)
    WD1 = weights3['wd1'].eval(sess)
    BD1 = weights3['bd1'].eval(sess)
    W_OUT = weights3['wout'].eval(sess)
    B_OUT = weights3['bout'].eval(sess)

# Create new graph for exporting
g = tf.Graph()
with g.as_default():
    x_2 = tf.placeholder("float", shape=[None, 784], name="input")

    WDC1 = tf.constant(WDC1, name="WDC1")
    WPC1 = tf.constant(WPC1, name="WPC1")

    x_image = tf.reshape(x_2, [-1, 28, 28, 1])
    CONV1 = separable_conv2d(x_image, WDC1, WPC1)
    MAXPOOL1 = maxpool2d(CONV1, k=2)

    WDC2 = tf.constant(WDC2, name="WDC2")
    WPC2 = tf.constant(WPC2, name="WPC2")
    CONV2 = separable_conv2d(MAXPOOL1, WDC2, WPC2)
    MAXPOOL2 = maxpool2d(CONV2, k=2)

    WD1 = tf.constant(WD1, name="WD1")
    BD1 = tf.constant(BD1, name="BD1")

    FC1 = tf.layers.flatten(MAXPOOL2)
    FC1 = tf.add(tf.matmul(FC1, WD1), BD1)
    FC1 = tf.nn.relu(FC1)

    W_OUT = tf.constant(W_OUT, name="W_OUT")
    B_OUT = tf.constant(B_OUT, name="B_OUT")

    # skipped dropout for exported graph as there is no need for already calculated weights

    OUTPUT = tf.nn.softmax(tf.matmul(FC1, W_OUT) + B_OUT, name="output")

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    graph_def = g.as_graph_def()
    tf.train.write_graph(graph_def, EXPORT_DIR, 'mnist_model_graph2.pb', as_text=False)

    # Test trained model
    y_train = tf.placeholder("float", [None, n_classes])
    correct_prediction = tf.equal(tf.argmax(OUTPUT, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("check accuracy %g" % accuracy.eval(
        {x_2: mnist.test.images, y_train: mnist.test.labels}, sess))
