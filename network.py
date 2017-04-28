
import tensorflow as tf
import re
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim

def hourglass_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """Defines the default ResNet arg scope.
  Args:
    is_training: Whether or not we are training the parameters in the batch
      normalization layers of the model.
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc



def hourglass(inputs,
            scale=1,
            regression_channels=2, classification_channels=22):
    """Defines a lightweight resnet based model for dense estimation tasks.
    Args:
      inputs: A `Tensor` with dimensions [num_batches, height, width, depth].
      scale: A scalar which denotes the factor to subsample the current image.
      output_channels: The number of output channels. E.g., for human pose
        estimation this equals 13 channels.
    Returns:
      A `Tensor` of dimensions [num_batches, height, width, output_channels]."""

    out_shape = tf.shape(inputs)[1:3]

    if scale > 1:
        inputs = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
        inputs = slim.layers.avg_pool2d(
            inputs, (3, 3), (scale, scale), padding='VALID')

    output_channels = regression_channels + classification_channels

    with slim.arg_scope(hourglass_arg_scope()):
        # D1
        net = slim.conv2d(inputs, 64, (7, 7), 2, scope='conv1')
        net = bottleneck_module(net, out_channel=128, res=128, scope='bottleneck1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # D2
        net = slim.stack(net, bottleneck_module, [(128, None),(128, None),(256, 256)], scope='conv2')

        # hourglasses (D3,D4,D5)
        with tf.variable_scope('hourglass'):
            net = hourglass_module(net, depth=3)

        # final layers (D6, D7)
        net = slim.stack(net, slim.conv2d, [(512, [1, 1]), (256, [1, 1]), 
                                            (output_channels, [1, 1])
                                           ], scope='conv3')

        net = tf.image.resize_bilinear(net, out_shape, name="up_sample")
        net = slim.conv2d(net, output_channels, 1, scope='conv_last')

    regression = slim.conv2d(net, regression_channels, 1, activation_fn=None)
    logits = slim.conv2d(net, classification_channels, 1, activation_fn=None)
    return regression, logits


def hourglass_module(inputs, depth=0):
    with tf.variable_scope('depth_{}'.format(depth)):
        # buttom up layers
        net = slim.max_pool2d(inputs, [2, 2], scope='pool')
        net = slim.stack(net, bottleneck_module, [(256,None),(256,None),(256,None)], scope='buttom_up')

        # connecting layers
        if depth > 0:
            net = hourglass_module(net, depth=depth-1)
        else:
            net = bottleneck_module(net, out_channel=512, res=512, scope='connecting')

        # top down layers
        net = bottleneck_module(net, out_channel=512, res=512, scope='top_down')
        net = tf.image.resize_bilinear(net, tf.shape(inputs)[1:3], name="up_sample")

        # residual layers
        net += slim.stack(inputs, bottleneck_module,
                          [(256, None),(256, None),(512, 512)], scope='res')

        return net

def bottleneck_module(inputs, out_channel=256, res=None, scope=''):

    with tf.variable_scope(scope):
        net = slim.stack(inputs, slim.conv2d, [(out_channel//2, [1, 1]), (out_channel//2, [3, 3]), (out_channel, [1, 1])], scope='conv')
        if res:
            inputs = slim.conv2d(inputs, res, (1, 1), scope='bn_res'.format(scope))
        net += inputs

        return net
