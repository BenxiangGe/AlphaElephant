import tensorflow as tf
import util

class CChessDataSet(object):
    def __init__(self, tf_record_files, batch_size=32):
        self.data_size = self.get_data_size(tf_record_files)

        filename_queue = tf.train.string_input_producer(tf_record_files)
        self.tfr_reader = tf.TFRecordReader()
        _, serialized_piece = self.tfr_reader.read(filename_queue)

        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image': tf.FixedLenFeature([], tf.string)
                  }

        self.data_features = tf.parse_single_example(serialized_piece, features=features)
        self.label = tf.cast(self.data_features['label'], tf.int32)
        self.label = tf.one_hot(self.label, util.BOARD_SIZE)

        self.image = tf.decode_raw(self.data_features['image'], tf.float64)
        self.image = tf.cast(self.image, tf.float32)
        self.image = tf.reshape(self.image, [util.Y_SIZE, util.X_SIZE, util.PIECE_SIZE])

        min_after_dequeue = 10
        # batch_size = 32
        self.capacity = min_after_dequeue + 3 * batch_size

        self.image_batch, self.label_batch = tf.train.shuffle_batch(
            [self.image, self.label], batch_size=batch_size, capacity=self.capacity, min_after_dequeue=min_after_dequeue)
        # self.image_batch, self.label_batch = tf.train.batch(
        #     [self.image, self.label], batch_size=batch_size, capacity=self.capacity)

    def get_data_size(self, tf_records_filenames):
        c = 0
        for file in tf_records_filenames:
            for record in tf.python_io.tf_record_iterator(file):
                c += 1
        return c

    def get_batch(self, session):
        return session.run([self.image_batch, self.label_batch])