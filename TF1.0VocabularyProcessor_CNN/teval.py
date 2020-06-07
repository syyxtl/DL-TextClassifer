#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1552893148/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

class TextCNN_:  
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'    #use GPU with ID=0
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
        self.config.gpu_options.allow_growth = True #allocate dynamically
        self.sess = tf.Session(config = self.config)
        self.vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.vocab_path)

    def dotextcnn(self,str):
        x_raw = [str]
        y_test = None
        x_test = np.array(list(self.vocab_processor.transform(x_raw)))
        print("\nEvaluating...\n")
        print(FLAGS.checkpoint_dir)
        # Evaluation
        # ==================================================
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
            # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
            # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
            # Collect the predictions here
                all_predictions = []
                for x_test_batch in batches:
                    batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                    if(batch_predictions[0] == 1):
                        return "positive"
                    else:
                        return "negative"

if __name__ == '__main__':
    tcnn = TextCNN_()
    tcnn.dotextcnn("this is a function and this is not a function")
    tcnn.dotextcnn("this is a function and this is not a function")