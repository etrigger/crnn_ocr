import tensorflow as tf
import numpy as np

num_classes = 993

num_features = 11
max_time_step = 512

hidden_units = 20


def inference(input_tensor, is_train=True):
    # input_size = [batch_size, width, height, channel]
    #            = [batch_size, 192, 32, 1]
    # output_size = [batch_size, 192, 32, 32]
    with tf.variable_scope('layer1-conv1'):
        conv1_w = tf.get_variable('weight', [3, 3, 1, 32], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        conv1_b = tf.get_variable('bias', [32], initializer=
        tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_w, strides=[1, 1, 1, 1],
                             padding='SAME')
        res1 = tf.nn.bias_add(conv1, conv1_b)
        if is_train:
            with tf.variable_scope('bn1'):
                mean1, variance1 = tf.nn.moments(res1, [0, 1, 2])
                scale1 = tf.Variable(tf.ones([32]))
                offset1 = tf.Variable(tf.zeros([32]))
                bn1 = tf.nn.batch_normalization(res1, mean1, variance1, offset1, scale1, 1e-3)
        relu1 = tf.nn.relu(bn1)

    # output_size = [batch_size, 96, 16, 32]
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    # output_size = [batch_size, 96, 16, 64]
    with tf.variable_scope('layer3-conv2'):
        conv2_w = tf.get_variable('weight', [3, 3, 32, 64], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        conv2_b = tf.get_variable('bias', [64], initializer=
        tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1],
                             padding='SAME')
        res2 = tf.nn.bias_add(conv2, conv2_b)
        if is_train:
            with tf.variable_scope('bn2'):
                mean2, variance2 = tf.nn.moments(res2, [0, 1, 2])
                scale2 = tf.Variable(tf.ones([64]))
                offset2 = tf.Variable(tf.zeros([64]))
                bn2 = tf.nn.batch_normalization(res2, mean2, variance2, offset2, scale2, 1e-3)
        relu2 = tf.nn.relu(bn2)

    # output_size = [batch_size, 48, 8, 64]
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    # output_size = [batch_size, 48, 8, 128]
    with tf.variable_scope('layer5-conv3'):
        conv3_w = tf.get_variable('weight', [3, 3, 64, 128], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        conv3_b = tf.get_variable('bias', [128], initializer=
        tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_w, strides=[1, 1, 1, 1],
                             padding='SAME')
        res3 = tf.nn.bias_add(conv3, conv3_b)
        if is_train:
            with tf.variable_scope('bn3'):
                mean3, variance3 = tf.nn.moments(res3, [0, 1, 2])
                scale3 = tf.Variable(tf.ones([128]))
                offset3 = tf.Variable(tf.zeros([128]))
                bn3 = tf.nn.batch_normalization(res3, mean3, variance3, offset3, scale3, 1e-3)
        relu3 = tf.nn.relu(bn3)

    # output_size = [batch_size, 24, 4, 128]
    with tf.variable_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    # output_size = [batch_size, 24, 4, 256]
    with tf.variable_scope('layer7-conv4'):
        conv4_w = tf.get_variable('weight', [3, 3, 128, 256], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        conv4_b = tf.get_variable('bias', [256], initializer=
        tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_w, strides=[1, 1, 1, 1],
                             padding='SAME')
        res4 = tf.nn.bias_add(conv4, conv4_b)
        if is_train:
            with tf.variable_scope('bn4'):
                mean4, variance4 = tf.nn.moments(res4, [0, 1, 2])
                scale4 = tf.Variable(tf.ones([256]))
                offset4 = tf.Variable(tf.zeros([256]))
                bn4 = tf.nn.batch_normalization(res4, mean4, variance4, offset4, scale4, 1e-3)
        relu4 = tf.nn.relu(bn4)

    # output_size = [batch_size, 12, 2, 256]
    with tf.variable_scope('layer8-pool4'):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    # output_size = [batch_size, 11, 1, 512]
    with tf.variable_scope('layer9-conv5'):
        conv5_w = tf.get_variable('weight', [2, 2, 256, 512], initializer=
        tf.truncated_normal_initializer(stddev=0.1))
        conv5_b = tf.get_variable('bias', [512], initializer=
        tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool4, conv5_w, strides=[1, 1, 1, 1],
                             padding='VALID')
        res5 = tf.nn.bias_add(conv5, conv5_b)
        if is_train:
            with tf.variable_scope('bn5'):
                mean5, variance5 = tf.nn.moments(res5, [0, 1, 2])
                scale5 = tf.Variable(tf.ones([512]))
                offset5 = tf.Variable(tf.zeros([512]))
                bn5 = tf.nn.batch_normalization(res5, mean5, variance5, offset5, scale5, 1e-3)
        relu5 = tf.nn.relu(bn5)
        # output_size = [batch_size, max_time_step, num_features]
        #             = [batch_size, 11, 512]
        seq = tf.transpose(relu5, (0, 3, 1, 2))
        seq = tf.reshape(seq, [-1, max_time_step, num_features])

    # output_size = [batch_size，max_time_step, hidden_units]
    with tf.variable_scope('layer10-blstm'):
        seq_len = np.ones([32]) * max_time_step
        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_units)
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_units)
        output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=seq, dtype=tf.float32)
        # output_size = [batch_size*max_time_step, hidden_units]
        output_reshaped = tf.reshape(output[0], [-1, hidden_units])

    with tf.variable_scope('layer11-affine_projection'):
        w = tf.get_variable('weight', [hidden_units, num_classes],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('bias', [num_classes], initializer=tf.constant_initializer(0.0))
        # output_size = [batch_size*max_time_step, num_classes]
        logits = tf.matmul(output_reshaped, w) + b
        # output_size = [batch_size, max_time_step, num_classes]
        logits = tf.reshape(logits, [-1, max_time_step, num_classes])
        # output_size = [max_time_step, batch_size, num_classes]
        logits = tf.transpose(logits, (1, 0, 2))

    return logits

