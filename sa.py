import tensorflow as tf
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
import sys

from common import *

class SA:
    def __init__(self, shape, no_classes, alpha=0.3, beta=0.7, learning_rate=0.001, batch_size=3, no_gpus=2, logs_path='./logs', save_path_models='./models', save_path_imgs = './images'):
        '''
        when change shape of input, need to change the first layer of Discriminator!!!
        '''
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.h, self.w, self.no_channels = shape
        self.no_classes = no_classes
        self.device = []
        for i in range(no_gpus):
            self.device.append('/device:GPU:{}'.format(i))
        # self.device.append('/cpu:0') 
        
        self.logs_path = logs_path
        self.save_path_models = save_path_models
        self.save_path_imgs = save_path_imgs
        self.istraining = tf.placeholder(tf.bool, None, 'istraining')
        self.x1 = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'img_1')
        self.x2 = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'img_2')
        self.x3 = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'img_3')
        self.learning_rate = {}
        for i in range(no_classes):
            # self.x1['class_{}'.format(i+1)] = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'class_{}_img_1'.format(i+1))
            # self.x2['class_{}'.format(i+1)] = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'class_{}_img_2'.format(i+1))
            # self.x3['class_{}'.format(i+1)] = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'class_{}_img_3'.format(i+1))
            self.learning_rate['class_{}'.format(i+1)] = tf.Variable(learning_rate, name='lr_{}'.format(i+1), trainable=False)
        self.y = tf.placeholder(tf.float32, [None, self.no_classes], 'labels')
        self.is_restored = False
    
    def generator(self, x=[], name=None):
        with tf.variable_scope(name+'_concat'):
            input_ = tf.concat(x, axis=3)
        with tf.variable_scope(name):
            net = tf.layers.conv2d(
                input_,
                16,
                [3,3],
                padding='same',
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                16,
                [5,5],
                padding='same',
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                32,
                [7,7],
                padding='same',
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                32,
                [5,5],
                padding='same',
                activation=tf.nn.leaky_relu,
            )
            net = tf.layers.conv2d(
                net,
                1,
                [3,3],
                padding='same',
                activation=tf.nn.tanh,
            )
        return net
    
    def classifier(self, x, istraining=False):
        with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
            net = block(x, 32, istraining=istraining, name='Block_1')
            net = block(net, 64, istraining=istraining, name='Block_2')
            net = block(net, 128, istraining=istraining, name='Block_3')
            net = block(net, 256, istraining=istraining, num_layers=3 ,name='Block_4')

            net = tf.layers.flatten(net)

            with tf.variable_scope('_dense_1'):
                net = tf.layers.dense(net, 256)
                net = tf.layers.batch_normalization(net, renorm=True, training=istraining)
                net = tf.nn.relu(net)
                net = tf.layers.dropout(net, training=istraining)

            with tf.variable_scope('_dense_2'):
                net = tf.layers.dense(net, 256)
                net = tf.layers.batch_normalization(net, renorm=True, training=istraining)
                net = tf.nn.relu(net)
                net = tf.layers.dropout(net, training=istraining)
            
            logits = tf.layers.dense(net, self.no_classes, activation=None)
            softmax = tf.nn.softmax(logits)
        return logits, softmax

    def build_model(self):
        generated_imgs = {}
        # loss_G = {}
        loss_all = {}
        var_list_G = {}
        summ_train = {}
        for i in range(self.no_classes):
            key = 'class_{}'.format(i+1)
            generated_imgs[key] = self.generator([self.x1, self.x2], name='Generator_{}'.format(key))
            with tf.variable_scope('concat_'+key):
                concat_imgs = tf.concat([generated_imgs[key], self.x3], axis=0)

            logits, softmax = self.classifier(concat_imgs, istraining=self.istraining)
            with tf.variable_scope("loss_" + key):
                loss_G = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.x3, predictions=generated_imgs[key]))
                # labels = one_hot(np.ones(self.batch_size), self.no_classes)
                loss_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))
                loss_all[key] = self.alpha * loss_G + self.beta * loss_C
            with tf.variable_scope("loss_summ_" + key):
                tf.summary.scalar('loss_G_{}'.format(key), loss_G, collections=[key])
                tf.summary.scalar('lr_{}'.format(key), self.learning_rate[key], collections=[key])
                tf.summary.scalar('loss_C_{}'.format(key), loss_C, collections=[key])
                tf.summary.scalar('loss_all_{}'.format(key), loss_all[key], collections=[key])
            var_list_G[key] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Generator_{}'.format(key))
            summ_train[key] = tf.summary.merge_all(key=key)
        self.summ_train = summ_train
        self.loss_all = loss_all
        self.generated_imgs = generated_imgs

        var_list_C = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Classifier')
        # for val - test - deploy
        with tf.variable_scope("deploy"):
            self.input_classify = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'input_C')
            logits, softmax = self.classifier(self.input_classify, istraining=False)
            self.loss_val = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits)
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(softmax,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.no_correct_predict = tf.reduce_sum(tf.cast(correct_prediction, "float"))

            self.loss_val_ph = tf.placeholder(tf.float32,  name='loss_val_ph')
            tf.summary.scalar('loss_val', self.loss_val_ph, collections=['val'])

            self.accu_val_ph = tf.placeholder(tf.float32,  name='accu_val_ph')
            tf.summary.scalar('accu_val', self.accu_val_ph, collections=['val'])
            self.summ_val = tf.summary.merge_all(key='val')
        
        self.optim = {}
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            for i in range(self.no_classes):
                key = 'class_{}'.format(i+1)
                with tf.variable_scope("optim_" + key):
                    self.optim[key] = tf.train.MomentumOptimizer(learning_rate=self.learning_rate[key], momentum=0.9, use_nesterov=True).minimize(loss_all[key], var_list=[var_list_G[key], var_list_C])
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver_models = tf.train.Saver(max_to_keep=10)
        print('build model successfully...')

        # with tf.Session() as sess:
        #     self.writer_train = tf.summary.FileWriter('./log')
        #     self.writer_train.add_graph(sess.graph)
        return
    
    def restore_all(self, path):
        '''
        restore all to continue training
        '''
        self.saver_models.restore(self.sess, path)
        self.is_restored = True
        return

    def random_class(self):
        return np.random.randint(0, self.no_classes, size=1)
    
    def random_batch(self, data):
        batch_size = self.batch_size
        class_id = np.random.randint(self.no_classes)
        class_size = data[class_id].shape[0]
        samples = np.random.randint(class_size, size=batch_size*3)
        batch_1 = data[class_id][samples[0:batch_size]]
        batch_2 = data[class_id][samples[batch_size:2*batch_size]]
        batch_3 = data[class_id][samples[2*batch_size:3*batch_size]]
        return batch_1, batch_2, batch_3, class_id


    def train(self, data,  data_val, labels_val, no_iteration=1000000000, global_summ=0, period_save_model = None):
        batch_size = self.batch_size
        no_data = data.shape[0]
        if not period_save_model:
            period_save_model = no_data
        print("Initiating...", period_save_model)

        if not self.is_restored:
            self.sess.run(tf.global_variables_initializer())

        self.writer_train = tf.summary.FileWriter(self.logs_path)
        if not os.path.isdir(self.save_path_models):
            os.makedirs(self.save_path_models)

        if not os.path.isdir(self.save_path_imgs):
            os.makedirs(self.save_path_imgs)

        self.writer_train.add_graph(self.sess.graph)
        loss_prev = {}
        loss_sum_10000_iter_current = {}
        local_iter = {}
        for i in range(self.no_classes):
            key = 'class_{}'.format(i+1)
            loss_prev[key] = 100000
            loss_sum_10000_iter_current[key] = []
            # after 100 iteration, compare loss, loss k giam thi se giam learning rate di 30%
            local_iter[key] = 0
        # ========== test ==========
        x_1_test = []
        x_2_test = []
        for i in range(self.no_classes):
            x_1_test.append(data[i][0:self.no_classes])
            x_2_test.append(data[i][self.no_classes:2*self.no_classes])
        x_1_test = np.concatenate(x_1_test, axis=0)
        x_2_test = np.concatenate(x_2_test, axis=0)
        print(x_1_test.shape)
        print(x_2_test.shape)
        img_test = (make_image_from_batch(x_1_test)*255).astype(np.uint8)
        cv2.imwrite('{}/parent_1.png'.format(self.save_path_imgs), img_test)
        img_test = (make_image_from_batch(x_2_test)*255).astype(np.uint8)
        cv2.imwrite('{}/parent_2.png'.format(self.save_path_imgs), img_test)
        # ========== test ==========
        print("Start training...")
        for iter_ in range(no_iteration):
            x_1_, x_2_, x_3_, class_id = self.random_batch(data)
            class_id = 0
            key = 'class_{}'.format(class_id+1)
            y_ = one_hot(np.ones(shape=self.batch_size*2)*class_id, self.no_classes)

            _ = self.sess.run(self.optim[key], feed_dict={self.x1: x_1_, self.x2: x_2_, self.x3: x_3_, self.y: y_, self.istraining: True})
            loss_curr, summ_train = self.sess.run( [self.loss_all[key], self.summ_train[key]], feed_dict={self.x1: x_1_, self.x2: x_2_, self.x3: x_3_, self.y: y_, self.istraining: False})
            loss_sum_10000_iter_current[key].append(loss_curr)
            self.writer_train.add_summary(summ_train, local_iter[key])

            if (local_iter[key] + 1) % 10000 == 0:
                loss_mean = np.mean(loss_sum_10000_iter_current[key])
                if loss_prev[key] <= loss_mean:
                    self.sess.run(tf.assign(self.learning_rate[key], self.learning_rate[key]*0.7))
                    print('decrease learning rate of  {}'.format(key))
                loss_prev[key] = loss_mean
                loss_sum_10000_iter_current[key] = []
                # compare loss, summary histogram
            local_iter[key] += 1
            # if i%5000==0:
            #     sys.stdout.write("\r  done: {}/{}".format(i, no_iteration))
            #     sys.stdout.flush()
            sys.stdout.write("\r  done >>>>>>>>>>>>>>: {}/{}".format(iter_, no_iteration))
            sys.stdout.flush()

            if (iter_+1) % period_save_model == 0:

                # create some image
                no_test_1 = x_1_test.shape[0]
                batch_generated_imgs = []
                for i in range(self.no_classes):
                    imgs = self.sess.run(self.generated_imgs['class_{}'.format(i+1)], {self.x1: x_1_test[i*7:(i+1)*7], self.x2: x_2_test[i*7:(i+1)*7]})
                    batch_generated_imgs.append(imgs)
                
                batch_generated_imgs = np.concatenate(batch_generated_imgs, axis=0)
                batch_generated_imgs = (make_image_from_batch(batch_generated_imgs)*255).astype(np.uint8)
                cv2.imwrite('{}/g_{}.png'.format(self.save_path_imgs, iter_), batch_generated_imgs)


                # create some image

                batch_val = 8
                no_samples_val = data_val.shape[0]
                loss_val = []
                no_correct_predict_val = []
                
                for i in range(no_samples_val//batch_val):
                    x_val_batch = data_val[batch_val*i:batch_val*(i+1)]
                    y_val_batch = labels_val[batch_val*i:batch_val*(i+1)]
                    loss_val_batch, no_correct_predict_batch = self.sess.run([self.loss_val, self.no_correct_predict], feed_dict={self.input_classify: x_val_batch, self.y: y_val_batch })
                    loss_val.append(loss_val_batch)
                    no_correct_predict_val.append(no_correct_predict_batch)

                x_val_batch = data_val[batch_val*(i+1):batch_val*(i+2)]
                y_val_batch = labels_val[batch_val*(i+1):batch_val*(i+2)]
                loss_val_batch, no_correct_predict_batch = self.sess.run([self.loss_val, self.no_correct_predict], feed_dict={self.input_classify: x_val_batch, self.y: y_val_batch })
                loss_val.append(loss_val_batch)
                no_correct_predict_val.append(no_correct_predict_batch)
                loss_val = np.concatenate(loss_val)
                loss_val = np.mean(loss_val)

                accu = np.sum(no_correct_predict_val)/no_samples_val
                print("")
                print('loss_val: {}'.format(loss_val))
                print('accu: {}'.format(accu))
                summ_val = self.sess.run(self.summ_val, feed_dict={self.loss_val_ph: loss_val, self.accu_val_ph: accu})
                self.writer_train.add_summary(summ_val, global_summ)
                save_path_models = self.saver_models.save(self.sess, "{}/model_{}/model.ckpt".format(self.save_path_models, global_summ))
                global_summ += 1
                print('saved model in: {}'.format(save_path_models))
                print("=======================================")
           




    

