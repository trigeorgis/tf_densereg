import tensorflow as tf
slim = tf.contrib.slim


def smooth_l1(pred, ground_truth, weights=1):
    """Defines a robust L1 loss.

    This is a robust L1 loss that is less sensitive to outliers
    than the traditional L2 loss. This was defined in Ross
    Girshick's Fast R-CNN, ICCV 2015 paper.

    Args:
      pred: A `Tensor` of dimensions [num_images, height, width, 3].
      ground_truth: A `Tensor` of dimensions [num_images, height, width, 3].
      weights: A `Tensor` of dimensions [num_images,] or a scalar with the
          weighting per image.
    Returns:
      A scalar with the mean loss.
    """
    residual = tf.abs(pred - ground_truth)

    loss = tf.where(tf.less(residual, 1),
                     0.5 * tf.square(residual),
                     residual - .5)

    return tf.losses.compute_weighted_loss(loss, weights)