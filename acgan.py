import tensorflow as tf
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt

from common import lrelu, one_hot, progress, make_image_from_batch

class ACGAN:
    # def __init__(self, data, labels, FLAGS):
    def __init__(self, shape, no_classes, learning_rate=0.0001, batch_size=100, no_gpus=2, logs_path='./log_ACGAN', save_path_models='./models_ACGAN', save_path_generator='./models_G', save_path_imgs = './images_ACGAN'):
        '''
        when change shape of input, need to change the first layer of Discriminator!!!
        '''
        self.batch_size = batch_size
        self.h, self.w, self.no_channels = shape
        self.no_classes = no_classes
        self.z_dim = no_classes + 100
        self.device = []
        for i in range(no_gpus):
            self.device.append('/device:GPU:{}'.format(i))
        # self.device.append('/cpu:0') 
        self.learning_rate = learning_rate
        self.logs_path = logs_path
        self.save_path_models = save_path_models
        self.save_path_generator = save_path_generator
        self.save_path_imgs = save_path_imgs
        self.is_restored = False
        self.istraining = tf.placeholder(tf.bool, None, 'istraining')

    def generator(self, Z, istraining = False, seperate_from_ACGAN = False):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        
        with tf.device(self.device[1]):
            with tf.variable_scope("Generator"):
                G_L1 = tf.layers.dense(Z, units = 384*6*6, activation = tf.nn.relu, kernel_regularizer=regularizer)
                
                G_L1 = tf.reshape(G_L1, (-1, 6, 6, 384))

                G_L2 = tf.layers.conv2d_transpose(
                    G_L1,
                    filters = 216,
                    kernel_size = [5, 5],
                    strides=(2, 2),
                    padding='same',
                    kernel_regularizer=regularizer
                )
                with tf.variable_scope("activation_G_L2"):
                    relu2 = tf.nn.relu(tf.layers.batch_normalization(G_L2, training=istraining))

                G_L3 = tf.layers.conv2d_transpose(
                    relu2,
                    filters = 108,
                    kernel_size = [5, 5],
                    strides=(2, 2),
                    padding='same',
                    kernel_regularizer=regularizer
                )
                with tf.variable_scope("activation_G_L3"):
                    relu3 = tf.nn.relu(tf.layers.batch_normalization(G_L3, training=istraining))

                G_L5 = tf.layers.conv2d_transpose(relu3, self.no_channels, [5, 5], strides=(2, 2), padding='same', kernel_regularizer=regularizer)
                G_out = tf.nn.tanh(G_L5)
                imgs_out = (G_out+1.0)/2.0
            if seperate_from_ACGAN:
                return (G_out+1.0)/2.0
            else:
                return G_out
# saver = tf.train.Saver({"v2": v2})
# save_path = self.saver_models.save(self.sess, "./models/model_{}/model.ckpt".format(epoch))
    # def save_G(self):
    #     # config = tf.ConfigProto(allow_soft_placement = True)
    #     # config.gpu_options.allow_growth = True
    #     # self.sess = tf.Session(config = config)
    #     saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator'))
    #     saver.save(self.sess, "./G_w/model.ckpt")
    #     self.sess.close()


    def discriminator(self, input, istraining, reuse = False):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        with tf.device(self.device[1]):
            with tf.variable_scope("Discriminator", reuse = reuse):
                with tf.variable_scope("block_0"):
                    D_0_drop = tf.layers.dropout(inputs=input, rate=0.5, training = istraining)
                    D_0 = tf.layers.conv2d(
                        inputs = D_0_drop,
                        filters = 16,
                        use_bias = True,
                        kernel_size = [3, 3],
                        strides = (2,2),
                        padding = "same",
                        name='D_0',
                        kernel_regularizer=regularizer
                    )
                    D_lrelu_0 = lrelu(D_0, 0.2)

                with tf.variable_scope("block_1"):
                    D_1_drop = tf.layers.dropout(inputs=D_lrelu_0, rate=0.5, training = istraining)
                    D_1 = tf.layers.conv2d(
                        inputs = D_1_drop,
                        filters = 32,
                        use_bias = True,
                        kernel_size = [3, 3],
                        strides = (1,1),
                        padding = "same",
                        name='D_1',
                        kernel_regularizer=regularizer
                    )
                    D_lrelu_1 = lrelu(tf.layers.batch_normalization(D_1, training=istraining), 0.2)
                
                with tf.variable_scope("block_2"):
                    D_2_drop = tf.layers.dropout(inputs=D_lrelu_1, rate=0.5, training = istraining)
                    D_2 = tf.layers.conv2d(
                        inputs = D_2_drop,
                        filters = 64,
                        use_bias = True,
                        kernel_size = [3, 3],
                        strides = (2,2),
                        padding = "same",
                        name='D_2',
                        kernel_regularizer=regularizer
                    )
                    D_lrelu_2 = lrelu(tf.layers.batch_normalization(D_2, training=istraining), 0.2)

                with tf.variable_scope("block_3"):
                    D_3_drop = tf.layers.dropout(inputs=D_lrelu_2, rate=0.5, training = istraining)
                    D_3 = tf.layers.conv2d(
                        inputs = D_3_drop,
                        filters = 128,
                        use_bias = True,
                        kernel_size = [3,3],
                        strides = (1,1),
                        padding = "same",
                        name='D_3',
                        kernel_regularizer=regularizer
                    )
                    D_lrelu_3 = lrelu(tf.layers.batch_normalization(D_3, training=istraining), 0.2)

                with tf.variable_scope("block_4"):
                    D_4_drop = tf.layers.dropout(inputs=D_lrelu_3, rate=0.5, training = istraining)
                    D_4 = tf.layers.conv2d(
                        inputs = D_4_drop,
                        filters = 256,
                        use_bias = True,
                        kernel_size = [3, 3],
                        strides = (2,2),
                        padding = "same",
                        name='D_4',
                        kernel_regularizer=regularizer
                    )
                    D_lrelu_4 = lrelu(tf.layers.batch_normalization(D_4, training=istraining), 0.2)

                with tf.variable_scope("block_5"):
                    D_5_drop = tf.layers.dropout(inputs=D_lrelu_4, rate=0.5, training = istraining)
                    D_5 = tf.layers.conv2d(
                        inputs = D_5_drop,
                        filters = 512,
                        use_bias = True,
                        kernel_size = [3, 3],
                        strides = (1,1),
                        padding = "same",
                        name='D_4',
                        kernel_regularizer=regularizer
                    )
                    D_lrelu_5 = lrelu(tf.layers.batch_normalization(D_5, training=istraining), 0.2)
                
                flatten = tf.layers.flatten(D_lrelu_5)

                with tf.variable_scope("real_or_fake"):
                    logits = tf.layers.dense(inputs = flatten, units = 1, kernel_regularizer=regularizer)
                    output = tf.nn.sigmoid(logits)
                with tf.variable_scope("classify_class"):
                    predict_class = tf.layers.dense(
                        inputs=flatten,
                        units=self.no_classes, 
                        name='fully_connected',
                        kernel_regularizer=regularizer
                    )
                    predict = tf.nn.softmax(predict_class, name='predict')
        return logits, output, predict_class
    def build_generator(self, seperate_from_ACGAN=False):
        dim_z = 100 + self.no_classes
        self.z = tf.placeholder(tf.float32, [None, 1, 1, dim_z], 'ph_Z') # noise + label
        self.g_net = self.generator(self.z, istraining= self.istraining, seperate_from_ACGAN=seperate_from_ACGAN)
        self.saver_G = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator'), max_to_keep=1)
        return
    
    def build_discriminator(self):
        self.X = tf.placeholder(tf.float32, [None, self.h, self.w, self.no_channels], 'ph_X')
        self.y = tf.placeholder(tf.int32, [None], 'ph_Y') # label 
        self.D_real_logits, self.D_real, self.predict_class_real = self.discriminator(self.X, self.istraining)
        self.D_fake_logits, self.D_fake, self.predict_class_fake = self.discriminator(self.g_net, self.istraining, reuse=True)
        return

    def build_model(self):
        self.build_generator()
        self.build_discriminator()

        # ===========
            # self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones([self.batch_size, 1])))
            # self.loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros([self.batch_size, 1])))
        self.D_loss_w = tf.reduce_mean(self.D_real_logits) - tf.reduce_mean(self.D_fake_logits)

        self.loss_Class_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_class_real, labels=self.y ))
        self.loss_Class_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_class_fake, labels=self.y ))  

        self.D_loss_all = self.loss_Class_fake + self.loss_Class_real + self.D_loss_w

        # self.loss_G_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones([self.batch_size, 1])))
        self.g_loss_w = tf.reduce_mean(self.D_fake_logits)
        self.G_loss_all = self.g_loss_w + self.loss_Class_fake

        l2_loss = tf.losses.get_regularization_loss()
        self.G_loss_all = self.G_loss_all + l2_loss
        self.D_loss_all = self.D_loss_all + l2_loss


        self.list_loss = [
            self.loss_Class_fake, self.loss_Class_real, self.D_loss_w,
            self.g_loss_w, self.D_loss_all,  self.G_loss_all
        ]

        #  ============ summary ================
        self.loss_G_w_ph = tf.placeholder(tf.float32, name='loss_g_w_ph')
        self.loss_G_total_ph = tf.placeholder(tf.float32, name='loss_G_total_ph')
        
        # self.loss_D_fake_ph = tf.placeholder(tf.float32, name='loss_d_fake_ph')
        self.loss_D_w_ph = tf.placeholder(tf.float32, name='loss_d_w_ph')
        self.loss_D_total_ph = tf.placeholder(tf.float32, name='loss_D_total_ph')
        
        self.loss_c_fake_ph = tf.placeholder(tf.float32, name='loss_c_fake_ph')
        self.loss_c_real_ph = tf.placeholder(tf.float32, name='loss_c_real_ph')

        with tf.variable_scope("loss_g"):
            tf.summary.scalar('loss_G_fake', self.loss_G_w_ph, collections=['loss'])
            tf.summary.scalar('loss_G_total', self.loss_G_total_ph, collections=['loss'])
        with tf.variable_scope("loss_d"):
            tf.summary.scalar('loss_D_fake', self.loss_D_w_ph, collections=['loss'])
            tf.summary.scalar('loss_D_total', self.loss_D_total_ph, collections=['loss'])
        with tf.variable_scope("loss_c"):
            tf.summary.scalar('loss_C_fake', self.loss_c_fake_ph, collections=['loss'])
            tf.summary.scalar('loss_C_real', self.loss_c_real_ph, collections=['loss'])

        self.summary_loss = tf.summary.merge_all(key='loss')
        # ===== optimize ====
        self.var_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator")

        self.var_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator")
        with tf.device(self.device[1]):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.D_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.D_loss_all, var_list=self.var_D)
                self.G_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.G_loss_all, var_list=self.var_G)

        self.d_clip = [tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)) for v in self.var_D]
        
        # ====

        config = tf.ConfigProto(allow_soft_placement=True)
        # config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        

        self.saver_models = tf.train.Saver(max_to_keep=2)
        
        # self.saver_models.restore(self.sess, "./model_332/model.ckpt")
        # self.saver_G.restore(self.sess, "./G_w/model.ckpt")
        
        print('build model successfully...')
        return
    
    def restore_all(self, path):
        '''
        restore all to continue training
        '''
        self.saver_models.restore(self.sess, path)
        self.is_restored = True
        print('Restore weight and bias of model successfully...')
        return
    
    def generate_by_class(self, class_, no_sample, sess=None):
        '''
        class_: label of class you want to create samples
        no_sample: number of samples you want to create
        '''
        labels = np.ones((no_sample)) * class_
        labels = one_hot(labels, depth = self.no_classes)
        z_ = np.random.normal(0, 1, (no_sample, 100))
        z_ = np.concatenate((z_, labels), axis = 1)
        z_ = z_.reshape((no_sample, 1, 1, -1))
        if sess==None:
            G_Images_c = self.sess.run(self.g_net, {self.z: z_, self.istraining: False})
        else:
            G_Images_c = sess.run(self.g_net, {self.z: z_, self.istraining: False})
        return G_Images_c
    def restore_generator(self, path, sess=None):
        '''
        restore only G to generate data
        '''
        if sess==None:
            self.saver_G.restore(self.sess, path)
        else:
            self.saver_G.restore(sess, path)
        print('Restore Generator successfully...')
        return

    def generate(self, labels, sess=None):
        '''
        class_: label of class you want to create samples
        no_sample: number of samples you want to create
        '''
        no_samples = labels.shape[0]
        labels = one_hot(labels, depth = self.no_classes)
        z_ = np.random.normal(0, 1, (no_samples, 100))
        z_ = np.concatenate((z_, labels), axis = 1)
        z_ = z_.reshape((no_samples, 1, 1, -1))
        if sess==None:
            G_Images_c = self.sess.run(self.g_net, {self.z: z_, self.istraining: False})
        else:
            G_Images_c = sess.run(self.g_net, {self.z: z_, self.istraining: False})
        return G_Images_c

    def train(self, data, labels, no_epochs=1000):
        batch_size = self.batch_size
        no_data = data.shape[0]
        bat_num = no_data // batch_size
        index_data = np.arange(no_data)
        print("Initiate...")

        if not self.is_restored:
            self.sess.run(tf.global_variables_initializer())

        self.writer_train = tf.summary.FileWriter(self.logs_path)
        if not os.path.isdir(self.save_path_models):
            os.makedirs(self.save_path_models)

        if not os.path.isdir(self.save_path_imgs):
            os.makedirs(self.save_path_imgs)

        if not os.path.isdir(self.save_path_generator):
            os.makedirs(self.save_path_generator)

        self.writer_train.add_graph(self.sess.graph)
        
        if self.no_classes > 10:
            labels_test = np.ones((10, 100), dtype=np.int16)
            for i in range(10):
                labels_test[i] = labels_test[i]*i
        else:
            labels_test = np.ones((self.no_classes, 100), dtype=np.int16)
            for i in range(self.no_classes):
                labels_test[i] = labels_test[i]*i

        z_test = np.random.normal(0, 1, (100, 100))
        generate_imgs_time = 0
        print("Start training ACGAN...")
        for epoch in range(no_epochs):
            print("")
            print('epoch {}:'.format(epoch+1))
            np.random.shuffle(index_data)
            start = time.time()
            x_ = []
            y_ = []
            z_ = []
            for ite in range(bat_num):
                x_ = data[index_data[ite*batch_size:(ite+1)*batch_size]]
                y_ = labels[index_data[ite*batch_size:(ite+1)*batch_size]]
                y_onehot = one_hot(y_, self.no_classes)
                z_ = np.random.normal(0, 1, (batch_size, 100))
                z_ = np.concatenate((z_, y_onehot), axis = 1)
                z_ = z_.reshape((batch_size, 1, 1, -1))

                if epoch==0:
                    self.sess.run(self.d_clip)
                    _ = self.sess.run(self.D_optim, {self.X: x_, self.y: y_, self.z: z_, self.istraining: True})
                    continue
                
                if (ite+1)%5==0:
                    # print('train g')
                    _ = self.sess.run(self.G_optim, {self.X: x_, self.y: y_, self.z: z_, self.istraining: True})
                else:
                    # print('train D')
                    self.sess.run(self.d_clip)
                    _ = self.sess.run(self.D_optim, {self.X: x_, self.y: y_, self.z: z_, self.istraining: True})

                if ite + 1 == bat_num: # every self.FLAGS.F_show_img batchs or final batch, we show some generated images
                    for i in range(labels_test.shape[0]):
                        # c means class
                        labels_test_c = one_hot(labels_test[i], self.no_classes)
                        no_test_sample = len(labels_test_c)
                        z_test_c = np.concatenate((z_test, labels_test_c), axis = 1)
                        z_test_c = z_test_c.reshape((no_test_sample, 1, 1, self.z_dim))

                
                        G_Images_c = self.sess.run(self.g_net, {self.z: z_test_c, self.istraining: False})
                        G_Images_c = (G_Images_c + 1.0) / 2.0
                        G_Images_c = make_image_from_batch(G_Images_c)
                        G_Images_c = (G_Images_c*255).astype(np.uint8)
                        if self.no_channels == 3:
                            G_Images_c = G_Images_c[:,:,::-1]
                        cv2.imwrite('{}/epoch_{}_class_{}.png'.format(self.save_path_imgs, epoch, i),G_Images_c)
                    generate_imgs_time = generate_imgs_time + 1  
                    labels_test_c = []
                progress(ite+1, bat_num)
            # we will show the loss of only final batch in each epoch
            # self.list_loss = [
            #     self.loss_Class_fake, self.loss_Class_real, self.D_loss_w,
            #     self.g_loss_w, self.D_loss_all,  self.G_loss_all
            # ]
            loss_C_fake, loss_C_real, D_loss_w, g_loss_w, D_loss_all, G_loss_all = self.sess.run(
                self.list_loss, 
                feed_dict = {
                   self.X: x_, 
                   self.y: y_, 
                   self.z: z_, 
                   self.istraining: False
                }
            )
            D_loss_all = D_loss_all/2.0  
            summary_for_loss = self.sess.run(
                self.summary_loss,
                feed_dict = { 
                    self.loss_c_fake_ph: loss_C_fake,
                    self.loss_c_real_ph: loss_C_real,
                    self.loss_D_w_ph: D_loss_w,
                    self.loss_D_total_ph: D_loss_all,
                    self.loss_G_w_ph: g_loss_w, 
                    self.loss_G_total_ph: G_loss_all, 
                    self.istraining: False
                }
            )
            self.writer_train.add_summary(summary_for_loss, epoch)
            save_path_models = self.saver_models.save(self.sess, "{}/model_{}/model.ckpt".format(self.save_path_models, epoch))

            save_path_G = self.saver_G.save(self.sess, "{}/model.ckpt".format(self.save_path_generator))

            stop = time.time()
            print("")
            print('time: {}'.format(stop - start))
            print('loss D: {}, loss G: {}'.format(D_loss_all, G_loss_all))
            print('saved model in: {}'.format(save_path_models))
            print('saved G in: {}'.format(save_path_G))
            print("")
            print("=======================================")
        self.writer_train.close()
        self.sess.close()







    
