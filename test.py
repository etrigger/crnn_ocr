import tensorflow as tf
import numpy as np
import utils
import random


path = '.\part2\\'
image_width = 192
image_height = 32

n_samples = 10072
max_len = 8

batch_size = 1


def next_batch(train_x, train_y, step, order):

    rnd_indices = order[batch_size*step:batch_size*(step+1)]
    
    x = []
    y = []
    
    for i in rnd_indices:
        x.append(train_x[i])
    batch_x = [np.reshape(xi, [image_width, image_height, 1]).astype(np.float32) for xi in x]

    for i in rnd_indices:
        y.append(train_y[i])
    batch_y = utils.sparse_tuple_from(y)

    return batch_x, batch_y


def main(_):
    char_dict = utils.get_char_dict(path)
    test_x, test_y = utils.get_data(path, char_dict)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.ckpt-40992.meta')
        saver.restore(sess, "./model/model.ckpt-40992")

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input/Placeholder:0')
        logits = graph.get_tensor_by_name('train/layer11-affine_projection/transpose:0')
        seq_len = np.ones([batch_size]) * max_len
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        order = list(range(0, n_samples))
        random.shuffle(order)

        sum = 0

        for step in range(n_samples):
            batch_x, batch_y = next_batch(test_x, test_y, step, order)
            feed_dict = {x: batch_x}
            pred = sess.run([decoded[0].values], feed_dict=feed_dict)

            gt = utils.get_char_res(char_dict, batch_y[1])
            print('Ground_Truth:{}'.format(gt))
            rr = utils.get_char_res(char_dict, pred[0])
            print('Recognize Result:{}'.format(rr))

            if gt == rr:
                sum += 1

        acc = sum/n_samples
        print('accuary:{}'.format(acc))


if __name__ == '__main__':
    tf.app.run()
