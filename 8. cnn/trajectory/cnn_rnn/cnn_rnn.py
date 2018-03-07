# 所需库包
import pandas as pd
import numpy as np
import tensorflow as tf
# 需要从我给的github上获得tfrecorder
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# %pylab inline

# mnist数据
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# 指定如何写成tfrecord文件的信息
# 每一个row是一个feature
df = pd.DataFrame({'name':['image','label'],
                  'type':['float32','int64'],
                  'shape':[(784,),()],
                  'isbyte':[False,False],
                  "length_type":['fixed','fixed'],
                  "default":[np.NaN,np.NaN]})
# 实例化该类
tfr = TFrecorder()

# 写训练集和测试集的位置
# mkdir mnist_tfrecord mnist_tfrecord/train mnist_tfrecord/test

# 用该方法写测试集的tfrecord文件
dataset = mnist.test
# 写在哪里
path = 'mnist_tfrecord/test/test'
# 把test_set写成符合要求的examples
test_set = []
for i in np.arange(dataset.num_examples):
    # 一个样本
    features = {}
    # 样本中的第一个feature
    features['image'] = dataset.images[i]
    # 样本中的第二个feature
    features['label'] = dataset.labels[i].astype('int64')
    test_set.append(features)
# 直接写入，每个tfrecord中写1000个样本
# 由于测试集里有10000个样本，所以最终会写出10个tfrecord文件
tfr.writer(path, test_set, num_examples_per_file = 1000)

# 结果：
# number of features in each example: 2
# 10000 examples has been written to mnist_tfrecord/test/test
# saved data_info to mnist_tfrecord/test/test.csv
    # name     type   shape isbyte length_type default
# 0  image  float32  (784,)  False       fixed     NaN
# 1  label    int64      ()  False       fixed     NaN
# 文件mnist_tfrecord/test/test.csv是后面导入tfrecord文件时会用到的信息

# 用该方法写训练集的tfrecord文件
dataset = mnist.train
path = 'mnist_tfrecord/train/train'
# 每个tfrecord文件写多少个样本
num_examples_per_file = 1000
# 当前写的样本数
num_so_far = 0
# 要写入的文件
writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, num_examples_per_file))
# 写多个样本
for i in np.arange(dataset.num_examples):
    # 要写到tfrecord文件中的字典
    features = {}
    # 写一个样本的图片信息存到字典features中
    tfr.feature_writer(df.iloc[0], dataset.images[i], features)
    # 写一个样本的标签信息存到字典features中
    tfr.feature_writer(df.iloc[1], dataset.labels[i], features)
    
    tf_features = tf.train.Features(feature= features)
    tf_example = tf.train.Example(features = tf_features)
    tf_serialized = tf_example.SerializeToString()
    writer.write(tf_serialized)
    # 每写了num_examples_per_file个样本就令生成一个tfrecord文件
    if i%num_examples_per_file ==0 and i!=0:
        writer.close()
        num_so_far = i
        writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, i+num_examples_per_file))
writer.close()
# 把指定如何写成tfrecord文件的信息保存起来
data_info_path = 'mnist_tfrecord/data_info.csv'
df.to_csv(data_info_path,index=False)

tfr = TFrecorder()
def input_fn_maker(path, data_info_path, shuffle=False, batch_size = 1, epoch = 1, padding = None):
    def input_fn():
        # tfr.get_filenames会返回包含path下的所有tfrecord文件的list
        # shuffle会让这些文件的顺序打乱
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset=tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle = shuffle, 
                            batch_size = batch_size, epoch = epoch, padding =padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn

# 这里的padding是可以用来给不同的特征进行reshape的
# 把image从原来的784维改成28*28的平面
padding_info = ({'image':[28,28,1],'label':[]})
# 生成3个input_fn
test_input_fn = input_fn_maker('mnist_tfrecord/test/',  'mnist_tfrecord/data_info.csv',
                            padding = padding_info)
train_input_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', shuffle=True, batch_size = 512,
                            padding = padding_info)
# 用来评估训练集用，不想要shuffle
train_eval_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', batch_size = 512,
                            padding = padding_info)
# input_fn在执行时会返回一个字典，里面的key对应着不同的feature(包括label在内)
# shape: [None,28,28,1]


def model_fn(features, mode):
    conv1 = tf.layers.conv2d(
        inputs=features['image'],
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv1')
    # shape: [None,28,28,32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name= 'pool1')
    # shape: [None,14,14,32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv2')
    # shape: [None,14,14,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name= 'pool2')
    # shape: [None,7,7,64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name= 'pool2_flat')
    # shape: [None,3136]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name= 'dense1')
    # dropout只在当mode为tf.estimator.ModeKeys.TRAIN时才使用
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # shape: [None,1024]
    logits = tf.layers.dense(inputs=dropout, units=10, name= 'output')
    # shape: [None,10]

    # shape: [None,28,28,1]
    # create RNN cells:
    rnn_cells = [tf.nn.rnn_cell.GRUCell(dim) for dim in [128,256]]
    # stack cells for multi-layers RNN
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
    # create RNN layers
    outputs, last_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=tf.reshape(features['image'],[-1,28,28]),
                                   dtype=tf.float32)
    # shape: outputs: [None,28,256]
    # shape: last_state: [None,256]
    dense1 = tf.layers.dense(inputs=last_state[1], units=1024, activation=tf.nn.relu, name= 'dense1')
    # shape: [None,1024]
   # 没用dropout
    # dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dense1, units=10, name= 'output')
    # shape: [None,10]

    # shape: [None,28,28,1]
    # create RNN cells:
    rnn_fcells = [tf.nn.rnn_cell.GRUCell(dim) for dim in [128,256]]
    rnn_bcells = [tf.nn.rnn_cell.GRUCell(dim) for dim in [128,256]]
    # stack cells for multi-layers RNN
    multi_rnn_fcell = tf.nn.rnn_cell.MultiRNNCell(rnn_fcells)
    multi_rnn_bcell = tf.nn.rnn_cell.MultiRNNCell(rnn_bcells)
    # create RNN layers
    ((outputs_fw, outputs_bw),(last_state_fw, last_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
                                   cell_fw=multi_rnn_fcell,
                                   cell_bw=multi_rnn_bcell,
                                   inputs=tf.reshape(features['image'],[-1,28,28]),
                                   dtype=tf.float32)
    # shape: outputs: [None,28,256]
    # shape: last_state: [None,256]
    dense1 = tf.layers.dense(inputs=last_state_fw[1]+last_state_bw[1], units=1024, activation=tf.nn.relu, name= 'dense1')
    # shape: [None,1024]
    # dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dense1, units=10, name= 'output')
    # shape: [None,10]

    # shape: [None,28,28,1]
    conv1 = tf.layers.conv1d(
            inputs = tf.reshape(features['image'],[-1,28,28]), 
            filters = 32, 
            kernel_size = 5,
            padding="same",
            activation=tf.nn.relu,
            name = 'conv1')
    # shape: [None,28,32]
    pool1 = tf.layers.max_pooling1d(inputs = conv1, 
                          pool_size=2,
                          strides=2,
                          name = 'pool1')
    # shape: [None,14,32]
    # create RNN cells:
    rnn_cells = [tf.nn.rnn_cell.GRUCell(dim) for dim in [128,256]]
    # stack cells for multi-layers RNN
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
    # create RNN layers
    outputs, last_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=pool1,
                                   dtype=tf.float32)
    # shape: outputs: [None,14,256]
    # shape: last_state: [None,256]
    dense1 = tf.layers.dense(inputs=last_state[1], units=1024, activation=tf.nn.relu, name= 'dense1')
    # shape: [None,1024]
    # dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dense1, units=10, name= 'output')
    # shape: [None,10]

    # 创建predictions字典，里面写进所有你想要在预测值输出的数值
    # 隐藏层的数值也可以，这里演示了输出所有隐藏层层结果。
    # 字典的key是模型，value给的是对应的tensor
    predictions = {
        "image":features['image'],
        "conv1_out":conv1,
        "pool1_out":pool1,
        "conv2_out":conv2,
        "pool2_out":pool2,
        "pool2_flat_out":pool2_flat,
        "dense1_out":dense1,
        "logits":logits,
        "classes": tf.argmax(input=logits, axis=1),
        "labels": features['label'],
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    # 当mode为tf.estimator.ModeKeys.PREDICT时，我们就让模型返回预测的操作
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

# 训练和评估时都会用到loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=features['label'], logits=logits)
    # 训练分支
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(
        loss=loss,
        # global_step用于记录训练了多少步
        global_step=tf.train.get_global_step())
        # 返回的tf.estimator.EstimatorSpec根据
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)        

# 注意评估的时候，模型和训练时一样，是一个循环的loop，不断累积计算评估指标。
    # 其中有两个局部变量total和count来控制
    # 把网络中的某个tensor结果直接作为字典的value是不好用的
    # loss的值是始终做记录的，eval_metric_ops中是额外想要知道的评估指标
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=features['label'], predictions=predictions["classes"])}
    # 不好用：eval_metric_ops = {"probabilities": predictions["probabilities"]}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# model_dir 表示模型要存到哪里
mnist_classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir="mnist_model_cnn")

    # 在训练或评估的循环中，每50次print出一次字典中的数值
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook])
# 训练集
eval_results = mnist_classifier.evaluate(input_fn=train_eval_fn, checkpoint_path=None)
print('train set')
print(eval_results)
# 测试集
# checkpoint_path是可以指定选择那个时刻保存的权重进行评估
eval_results = mnist_classifier.evaluate(input_fn=test_input_fn, checkpoint_path=None)
print('test set')
print(eval_results)

predicts =list(mnist_classifier.predict(input_fn=test_input_fn))
predicts[0].keys()
# 输出为：
dict_keys(['image', 'conv1_out', 'pool1_out', 'conv2_out', 'pool2_out', 'pool2_flat_out', 'dense1_out', 'logits', 'classes', 'labels', 'probabilities'])

plt.figure(num=4,figsize=(28,28))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(predicts[0]['conv1_out'][:,:,i],cmap = plt.cm.gray)
plt.savefig('conv1_out.png')

# tensorboard --logdir=mnist_model_cnn
# 打开http://localhost:6006         