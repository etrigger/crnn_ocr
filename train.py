import utils
import crnn_1
import tensorflow as tf
import numpy as np
import random

path = '.\part2\\'
image_width = 192
image_height = 32

n_samples = 10072
max_len = 8

batch_size = 8
num_epoch = 20
learning_rate = 0.01


def next_batch(train_x, train_y, step, order):

    rnd_indices = order[batch_size*step:batch_size*(step+1)]
    x = []
    y = []
    for i in rnd_indices:
        x.append(train_x[i])
    # batch_x = x
    batch_x = [np.reshape(xi, [image_width, image_height, 1]).astype(np.float32) for xi in x]
    # start = step * batch_size
    # end = (step + 1) * batch_size - 1
    # x = train_x[start:end]
    # batch_x = [np.reshape(xi, [image_width, image_height, 1]).astype(np.float32) for xi in x]
    # y = train_y[start:end]
    for i in rnd_indices:
        y.append(train_y[i])
    batch_y = utils.sparse_tuple_from(y)

    return batch_x, batch_y


def train(char_dict, train_x, train_y):
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, image_width, image_height, 1])
        # reshaped_x = tf.reshape(x, [None, image_width, image_height, 1])
        targets = tf.sparse_placeholder(tf.int32)
        seq_len = np.ones([batch_size])*max_len

    with tf.name_scope('train'):
        logits = crnn_1.inference(x)
        with tf.name_scope('cost'):
            loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
            cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
        tf.summary.scalar('cost', cost)

    # with tf.name_scope('acc'):
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    #     with tf.name_scope('accuracy'):
    #     # acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    #     # gt = tf.cast(tf.sparse_tensor_to_dense(targets), tf.int32)
    #     # rec = tf.cast(tf.sparse_tensor_to_dense(decoded[0]), tf.int32)
    #         acc = utils.cal_rec_acc(targets, decoded[0])
    #     tf.summary.scalar('accuracy', acc)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('./log', sess.graph)
        sess.run(init)

        order = list(range(0, n_samples))
        random.shuffle(order)

        for curr_epoch in range(num_epoch):
            print("Epoch.......", curr_epoch)
            # null_res = 0
            for step in range(1258):
                batch_x, batch_y = next_batch(train_x, train_y, step, order)
                feed_dict = {x: batch_x, targets: batch_y}
                summary, cost_, _, g_step, label, rec = sess.run([merged, cost, optimizer, global_step, targets.values, decoded[0].values], feed_dict=feed_dict)
                # print("rec.shape:".format(np.shape(dec)))
                # if (len(rec) < batch_size*2):
                #     break
                gt = utils.get_char_res(char_dict, label)
                rr = utils.get_char_res(char_dict, rec)

                print("step:{}, train_cost = {:.3f}".format(step, cost_))
                # print("rec.size:{}".format(len(rec)))
                print('Ground_Truth:{}'.format(gt))
                print('Recognize Result:{}'.format(rr))
                # if len(rr) == 0:
                #     null_res += 1
                # if null_res > 100:
                #     break
                summary_writer.add_summary(summary, g_step)

        summary_writer.close()
        saver.save(sess, "./model/model.ckpt")


def main(argv=None):
    char_dict = utils.get_char_dict(path)
    train_x, train_y = utils.get_data(path, char_dict)
    train(char_dict, train_x, train_y)


if __name__ == '__main__':
    tf.app.run()
