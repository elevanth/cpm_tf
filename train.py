import os
import cv2
import numpy as np
import tensorflow as tf

import cpm_model
import tf_utils, cpm_utils

'''
Parameters
'''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tfr_data_dir',
                           default_value=['./inputs/cpm_aichallenger_dataset.tfrecords'],
                           docstring='Training data tfrecords')
tf.app.flags.DEFINE_string('pretrained_model',
                           default_value='./models',
                           docstring='pretrained mode')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=128,
                            docstring='input image size')
tf.app.flags.DEFINE_integer('heatmap_size',
                            default_value=32,
                            docstring='output heatmap size')
tf.app.flags.DEFINE_integer('stages',
                            default_value=6,
                            docstring='CPM stages number')
tf.app.flags.DEFINE_integer('joints_num',
                            default_value=14,
                            docstring='joints number')
tf.app.flags.DEFINE_integer('batch_size',
                            default_value=32,
                            docstring='training batch size')
tf.app.flags.DEFINE_integer('training_iterations',
                            default_value=300000,
                            docstring='max training iterations')
tf.app.flags.DEFINE_float('learning_rate',
                          default_value=0.0001,
                          docstring='initial learning rate')
tf.app.flags.DEFINE_float('lr_decay_rate',
                          default_value=0.9,
                          docstring='learning rate decay rate')
tf.app.flags.DEFINE_integer('lr_decay_step',
                          default_value=2000,
                          docstring='learning rate decay steps')
tf.app.flags.DEFINE_string('saved_model_path',
                           default_value='cpm_i%s*%s_o%s*%s_s%s',
                           docstring='saved model path')
tf.app.flags.DEFINE_string('saved_log_path',
                           default_value='cpm_i%s*%s_o%s*%s_s%s',
                           docstring='saved log path')
tf.app.flags.DEFINE_string('saved_name',
                           default_value='cpm_i`%s*%s_o%s*%s_s%s',
                           docstring='saved results name mode')


def main(argv):
    result_name = FLAGS.saved_log_path % (str(FLAGS.input_size), str(FLAGS.input_size),
                                      str(FLAGS.heatmap_size), str(FLAGS.heatmap_size), str(FLAGS.stages))
    trained_model_dir = os.path.join('.', 'results', 'models', result_name)
    log_dir = os.path.join('.', 'results', 'logs', result_name)

    min_loss = 300

    batch_x, batch_y, batch_x_ori = tf_utils.read_batch_cpm(FLAGS.tfr_data_dir, FLAGS.input_size,
                                                            FLAGS.joints_num, batch_size=FLAGS.batch_size)
    input_x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 3),
                             name='input_x')
    input_y = tf.placeholder(dtype=tf.float32,
                             shape=(FLAGS.batch_size, FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.joints_num),
                             name='input_heatmap')

    model = cpm_model.CPM_Model(FLAGS.stages, FLAGS.joints_num+1)
    model.build_model(input_x, FLAGS.batch_size)
    model.build_loss(input_y, FLAGS.learning_rate, FLAGS.lr_decay_rate, FLAGS.lr_decay_step)
    print('====Model Build====\n')

    '''Training'''
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tf_writer = tf.summary.FileWriter(log_dir, sess.graph, filename_suffix=result_name)

        saver = tf.train.Saver(max_to_keep=None)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        if FLAGS.pretrained_model is not None:
            saver.restore(sess, FLAGS.pretrained_model)

            for variable in tf.trainable_variables():
                with tf.variable_scope('print weights', reuse=True):
                    var = tf.get_variable(variable.name.split(':0')[0])
                    print(variable.name, np.mean(sess.run(var)))

        while True:
            batch_x_tf, batch_y_tf = sess.run([batch_x, batch_y])

            # warp training images
            for img_num in range(batch_x_tf.shape[0]):
                deg1 = (2 * np.random.rand() - 1) * 50
                deg2 = (2 * np.random.rand() - 1) * 50
                batch_x_tf[img_num, ...] = cpm_utils.warpImage(batch_x_tf[img_num, ...], 0, deg1, deg2, 1, 30)
                batch_y_tf[img_num, ...] = cpm_utils.warpImage(batch_y_tf[img_num, ...], 0, deg1, deg2, 1, 30)
                batch_y_tf[img_num, :, :, FLAGS.joints_num] = np.ones(shape=(FLAGS.input_size, FLAGS.input_size))\
                                                              - np.max(batch_y_tf[img_num, :, :, FLAGS.joints_num],
                                                                       axis=2)
            # Recreate heatmaps
            heatmaps_tf = cpm_utils.make_gaussian_batch(batch_y_tf, FLAGS.heatmap_size, 3)

            stage_losses, total_loss, _, summary, current_lr, stage_heatmaps, global_step =\
                sess.run([model.stage_loss,
                          model.total_loss,
                          model.train_op,
                          model.merged_summary,
                          model.lr,
                          model.stage_heatmaps,
                          model.global_step],
                         feed_dict={input_x: batch_x_tf,
                                    input_y: heatmaps_tf})

            tf_writer.add_summary(summary, global_step)

            if global_step%50 == 0:
                demo_img = batch_x_tf[0] + 0.5
                demo_stage_heatmaps = []
                for stage in range(FLAGS.stages):
                    demo_stage_heatmap = stage_heatmaps[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
                        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                    demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
                    demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
                    demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                    demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
                    demo_stage_heatmaps.append(demo_stage_heatmap)

                demo_img_heatmap = heatmaps_tf[0, :, :, 0:FLAGS.num_of_joints].reshape(
                    (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
                demo_img_heatmap = cv2.resize(demo_img_heatmap, (FLAGS.input_size, FLAGS.input_size))
                demo_img_heatmap = np.amax(demo_img_heatmap, axis=2)
                demo_img_heatmap = np.reshape(demo_img_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
                demo_img_heatmap = np.repeat(demo_img_heatmap, 3, axis=2)

                if FLAGS.stages > 4:
                    upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]),
                                               axis=1)
                    blend_img = 0.5 * demo_img_heatmap + 0.5 * demo_img
                    lower_img = np.concatenate((demo_stage_heatmaps[FLAGS.stages - 1], demo_img_heatmap, blend_img),
                                               axis=1)
                    demo_img = np.concatenate((upper_img, lower_img), axis=0)
                    cv2.imshow('current heatmap', (demo_img * 255).astype(np.uint8))
                    cv2.waitKey(1000)
                else:
                    upper_img = np.concatenate((demo_stage_heatmaps[FLAGS.stages - 1], demo_img_heatmap, demo_img),
                                               axis=1)
                    cv2.imshow('current heatmap', (upper_img * 255).astype(np.uint8))
                    cv2.waitKey(1000)

            print('##========Iter {:>6d}========##'.format(global_step))
            print('Current learning rate: {:.8f}'.format(current_lr))
            for stage_num in range(FLAGS.stages):
                print('Stage {} loss: {:>.3f}'.format(stage_num + 1, stage_losses[stage_num]))
            print('Total loss: {:>.3f}\n\n'.format(total_loss))

            # Save models
            if total_loss <= min_loss:
                min_loss = total_loss
                saver.save(sess=sess, save_path=trained_model_dir, global_step=global_step)
                print('\nModel checkpoint saved...\n')

            # Finish training
            if global_step == FLAGS.training_iterations:
                break

        coord.request_stop()
        coord.join(threads)

    print('Training done.')



