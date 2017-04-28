import tensorflow as tf
import numpy as np

from pathlib import Path

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    # RGB -> BGR
    image = tf.reverse(image, [2])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image


class Dataset:
    def __init__(self, names, batch_size=8, is_training=False):
        self.batch_size = batch_size
        self.root = Path('/vol/atlas/databases/tf_records/')
        self.tfrecord_names = names
        self.is_training = is_training
        
    def get(self):
        paths = [str(self.root / x) for x in self.tfrecord_names]

        filename_queue = tf.train.string_input_producer(paths)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'uv': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
            })

        image = tf.image.decode_jpeg(features['image'])
        image = tf.to_float(image)
        
        height, width = 600, 600
        
        image.set_shape((height, width, 3))
        image = caffe_preprocess(image)
        
        uv = tf.decode_raw(features['uv'], tf.float32)
        
        uv.set_shape((3 * width * height))
        uv = tf.reshape(uv, (height, width, 3))
        
        if self.is_training:
            image = distort_color(image / 255.) * 255.
            
        mask = tf.to_float(tf.reduce_mean(uv, 2) >= 0)[..., None]
        
        return tf.train.shuffle_batch(
            [image, uv[..., :2], mask],
            self.batch_size,
            capacity=1000,
            num_threads=2,
            min_after_dequeue=200)


class SyntheticDataset(Dataset):
    def __init__(self, **kwargs):
        names = ['synthetic_densereg_600x600.tfrecords']
        super().__init__(names, **kwargs)
    
    def num_samples(self):
        return 14193 * 3

class Dataset300W(Dataset):
    def __init__(self, **kwargs):
        names = ['300w_densereg_600x600.tfrecords']
        super().__init__(names, **kwargs)
    
    def num_samples(self):
        return 600

class DatasetMixer():
    def __init__(self, names, densities=None, batch_size=1):
        self.providers = []
        self.batch_size = batch_size
        
        if densities is None:
            densities = [1] * len(names)

        for name, bs in zip(names, densities):
            provider = globals()[name](batch_size=bs)
            self.providers.append(provider)
            
    def get(self, **kargs):
        queue = None
        enqueue_ops = []
        for p in self.providers:
            tensors = p.get(**kargs)
            
            shapes = [x.get_shape() for x in tensors]

            if queue is None:
                dtypes = [x.dtype for x in tensors]
                queue = tf.FIFOQueue(
                    capacity=1000,
                    dtypes=dtypes, name='fifoqueue')

            enqueue_ops.append(queue.enqueue_many(tensors))

        qr = tf.train.QueueRunner(queue, enqueue_ops)
        tf.train.add_queue_runner(qr)      

        tensors = queue.dequeue()
        
        for t, s in zip(tensors, shapes):
            t.set_shape(s[1:])

        return tf.train.batch(
            tensors,
            self.batch_size,
            num_threads=2,
            enqueue_many=False,
            dynamic_pad=True,
            capacity=200)


    