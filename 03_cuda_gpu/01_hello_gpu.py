# #TF1.x hello world:
# import tensorflow as tf
# msg = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(msg))

#https://stackoverflow.com/questions/55142951/tensorflow-2-0-attributeerror-module-tensorflow-has-no-attribute-session
# TF 2.0
import tensorflow as tf
msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))