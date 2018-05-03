import tensorflow as tf
import numpy as np



# Training parameters:
learning_rate = 0.01
num_steps = 100
xPool_batch_size = 100

# other param
ae_display_step = 1
sl_display_step = 10

# Producer for X_pool
def next_batch(start = 0):
    next_start = start + xPool_batch_size
    return X_pool[start:start + xPool_batch_size], next_start

# Producer for training data
def next_sup_train_entry(index = 0):
    x = Mat_Label[index:index+sl_display_step]
    y = labels[index:index+sl_display_step]
    print(y.shape)
    index = index + sl_display_step
    return x, y, index

# Autoencoder Parameters:
# TODO: Tune the Following Network parameters:
num_input = sonyc_length # Fixed
num_hidden_1 = 64 # 1st layer num features
num_hidden_2 = 32 # 2nd layer num features (the latent dim)
num_hidden_3 = 24

X = tf.placeholder(tf.float32, [None, num_input])
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    return layer_3

# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    return layer_3

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Define loss and optimizer, minimize the squared error
y_pred = decoder_op
y_true = X
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Second Part:
# Construct the supervised learning model after autoencoder
W = tf.Variable(tf.zeros([num_hidden_3, class_length]))   # Connect the layer directly after the encoder
b = tf.Variable(tf.zeros([class_length]))

# define the one FC layer softmax
pred_y = tf.nn.softmax(tf.matmul(encoder_op, W) + b)
# correct label holder
true_y = tf.placeholder(tf.int32, [None])

loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_y, logits=pred_y))
optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_2, var_list=[W, b])

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    start = 0
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, _ = mnist.train.next_batch(batch_size)

        batch_x, start = next_batch(start)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x/256})
        # Display logs per step
        if i % ae_display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))


    # Now the encoder has its parameters trained.
    # Take the encoder and add layers of NN and FC for supervised learning on labeled data.
    index = 0
    for t in range(1, Mat_Label.shape[0]):
        this_x, this_true_y, index = next_sup_train_entry(index)
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        _, l = sess.run(optimizer_2, feed_dict={X:this_x/256, true_y:this_true_y})
        # if t % sl_display_step == 0 or t == 1:
        print('Step %i: Loss: %f' % (t, l))


    #test: display something
    # index = 0
    # for i in range(5):
    #     this_x, this_true_y, index = next_sup_train_entry(index)
    #     g = sess.run(decoder_op, feed_dict={X: this_x})
    #
    #     print('input x:')
    #     print(this_x)
    #     print('output g'+str(g.shape))
    #     print(g*256)
    #

writer.close()