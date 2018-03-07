


import tensorflow as tf
def build_rnn():
    
def build_cnn(features, mode):
    
    image_batch = features['x']
    
    with tf.name_scope("conv1"):  
        conv1 = tf.layers.conv2d(inputs=image_batch, filters=32, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.relu)

    with tf.name_scope("pool1"):  
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.name_scope("conv2"):  
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.relu)

    with tf.name_scope("pool2"):  
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.name_scope("dense"):  
        # The 'images' are now 7x7 (28 / 2 / 2), and we have 64 channels per image
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    with tf.name_scope("dropout"):  
        # Add dropout operation; 0.8 probability that a neuron will be kept
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.2, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)


def model_fn(features, labels, mode):
    
    logits = build_cnn(features, mode)
    
    # Generate Predictions
    classes = tf.argmax(logits, axis=1)
    predictions = {
        'classes': classes,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return an EstimatorSpec for prediction
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
    # Compute the loss, per usual.
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)
        
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        # Configure the Training Op
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=1e-3,
            optimizer='Adam')

        # Return an EstimatorSpec for training
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                      loss=loss, train_op=train_op)    

    assert mode == tf.estimator.ModeKeys.EVAL
    
    # Configure the accuracy metric for evaluation
    metrics = {'accuracy': tf.metrics.accuracy(classes, tf.argmax(labels, axis=1))}
    
    return tf.estimator.EstimatorSpec(mode=mode, 
                                      predictions=predictions, 
                                      loss=loss,
                                      eval_metric_ops=metrics)    


