#！/usr/bin/env.python
# _*_ coding:utf-8 _*_


import os

import input_data
import model
import tensorflow as tf

#下载数据集
data = input_data.read_data_sets('MNIST_data', one_hot=True)

#创建模型
with tf.variable_scope('regression'):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)


# 定义训练相关指标
y_ = tf.placeholder('float', [None, 10])
# 定义交叉熵, 即代表loss
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 使用梯度下降的优化器，步长0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 比较label和predicts
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 把correct_prediction返回的bool类型转化为浮点型，取平均值得到准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver() # 保存以上定义的变量
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # 变量初始化

    # 开始训练
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        # feed_dict 是给使用placeholder创建的参数赋值
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print(sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_: data.test.labels}))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False
    )
    print('Saved:', path)
