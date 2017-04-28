import tensorflow as tf
import numpy as np
import network
import losses
import data_provider

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('batch_size', 8, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many preprocess threads to use.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0',
                           '''Device to train with.''')
tf.app.flags.DEFINE_integer('quantization_step', 10,
                            '''The quantization step size.''')
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def train():
    g = tf.Graph()
    with g.as_default():
        # Load datasets.
        names = ['SyntheticDataset', 'Dataset300W']
        provider = data_provider.DatasetMixer(
            names, batch_size=FLAGS.batch_size, densities=(1, 60))
        
        images, uvs, masks = provider.get()
        
        images = tf.image.resize_images(images, (200, 200), method=0)
        uvs = tf.image.resize_images(uvs, (200, 200), method=1)
        masks = tf.image.resize_images(masks, (200, 200), method=1)
        
        # Define model graph.
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                            is_training=True):
            prediction, logits = network.hourglass(images)

        loss = losses.smooth_l1(prediction, uvs)
        tf.summary.scalar('smooth l1 loss', loss)

        k = FLAGS.quantization_step
        n_classes = k + 1
        
        for i, name in enumerate(['hor', 'ver']):
            gt = tf.to_int64(tf.floor(uvs[..., i] * k))
            gt = tf.reshape(gt, [-1])
            gt = slim.one_hot_encoding(gt, n_classes)
            class_loss = tf.contrib.losses.softmax_cross_entropy(
                tf.reshape(logits[..., i * n_classes: (i+1) * n_classes], [-1, n_classes]), gt)
            tf.summary.scalar('losses/classification loss [{}]'.format(name), class_loss)                
        
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)

        tf.summary.image('images', images)
        tf.summary.image('uvs/h', uvs[..., :1])
        tf.summary.image('uvs/v', uvs[..., 1:])
        tf.summary.image('predictions/h', prediction[..., :1])
        tf.summary.image('predictions/v', prediction[..., 1:])

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    config = tf.ConfigProto(inter_op_parallelism_threads=2)
    
    with tf.Session(graph=g, config=config) as sess:
        init_fn = None

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading pretrained model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                    FLAGS.pretrained_model_checkpoint_path,
                    variables_to_restore, ignore_missing_vars=True)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)


if __name__ == '__main__':
    train()
