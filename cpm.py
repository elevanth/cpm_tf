import os
import cv2
import math
import time
import imageio
import numpy as np
import tensorflow as tf

import cpm_model
import cpm_utils


'''
Parameters
'''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           default_value='image',
                           docstring='input data type image / video')
tf.app.flags.DEFINE_string('input_dir',
                           default_value='./dataset',
                           docstring='directory of input data')
tf.app.flags.DEFINE_string('model_path',
                           default_value='models/cpm_body.pkl',
                           docstring='trained model')
tf.app.flags.DEFINE_integer('batch_size',
                            default_value=32,
                            docstring='input image size')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=368,
                            docstring='input image size')
tf.app.flags.DEFINE_integer('heatmap_size',
                            default_value=46,
                            docstring='output heatmap size')
tf.app.flags.DEFINE_integer('centermap_radius',
                            default_value=21,
                            docstring='center map gaussian variance')
tf.app.flags.DEFINE_integer('joints',
                            default_value=14,
                            docstring='number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default_value=6,
                            docstring='cpm stage number')
# tf.app.flags.DEFINE_string('color_channel',
#                            default_value='RGB',
#                            docstring='input image color channel')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default_value=21,
                            docstring='Center map gaussian variance')

# Set color for each finger
JOINT_COLOR_CODE = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]


LIMB_CONNECT = [[0, 1],
                [2, 3],
                [3, 4],
                [5, 6],
                [6, 7],
                [8, 9],
                [9, 10],
                [11, 12],
                [12, 13]]

def main(argv):
    tf_device = '/gpu:0'
    with tf.device(tf_device):
        '''
        Bulid graph
        '''
        input_data = tf.placeholder(dtype=tf.float32,
                                    shape=[None, FLAGS.input_size, FLAGS.input_size, 3],
                                    name='input_image')
        center_map = tf.placeholder(dtype=tf.float32,
                                    shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                    name='center_map')
        model = cpm_model.CPM_Model(FLAGS.stages, FLAGS.joints+1)
        model.build_model(input_data, center_map, FLAGS.batch_size)

    saver = tf.train.Saver()

    test_center_map = cpm_utils.gaussian_img(FLAGS.input_size,
                                             FLAGS.input_size,
                                             FLAGS.input_size / 2,
                                             FLAGS.input_size / 2,
                                             FLAGS.cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, FLAGS.input_size,
                                                   FLAGS.input_size, 1])

    if FLAGS.DEMO_TYPE == 'image':
        all_files = os.walk(FLAGS.input_dir)
        for root, dirs, files in all_files:
                input_flow = [os.path.join(FLAGS.input_dir, file) for file in files]
    else:
        input_flow = imageio.get_reader(FLAGS.input_dir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # saver.restore(sess, FLAGS.model_path)
        model.load_weights_from_file(FLAGS.model_path, sess, False)

        with tf.device(tf_device):
            # tic = time.time()

            for i, im in enumerate(input_flow):
                if im.split('/')[-1].startswith('.'):
                    continue
                img_tic = time.time()
                test_img = cpm_utils.read_image(im, FLAGS.input_size)

                test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))
                print('img read time %f' % (time.time() - img_tic))

                test_img_input = test_img_resize / 256.0 - 0.5
                test_img_input = np.expand_dims(test_img_input, axis=0)

                fps_tic = time.time()
                predict_heatmap, stage_heatmaps = sess.run([model.output_heatmap,
                                                           model.stage_heatmaps],
                                                          feed_dict={input_data: test_img_input,
                                                                     'center_map:0': test_center_map})
                demo_img = visualize_result(test_img, FLAGS, stage_heatmaps)
                cv2.imshow('demo_img', demo_img.astype(np.uint8))
                if cv2.waitKey(0) == ord('q'): exit()
                print('fps: %.2f' % (1 / (time.time() - fps_tic)))


def visualize_result(test_img, FLAGS, stage_heatmaps):
    last_heatmap = stage_heatmaps[-1][0, :, :, 0:FLAGS.joints].reshape(
        FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.joints)
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    print('')

    joint_coord_set = np.zeros((FLAGS.joints, 2))

    for joint_num in range(FLAGS.joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)
        joint_color = list(map(lambda x: x + 35 * (joint_num % 4),
                               JOINT_COLOR_CODE[color_code_num]))
        cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3,
                   color=joint_color, thickness=-1)

    # Plot limb colors
    for limb_num in range(len(LIMB_CONNECT)):

        x1 = joint_coord_set[LIMB_CONNECT[limb_num][0], 0]
        y1 = joint_coord_set[LIMB_CONNECT[limb_num][0], 1]
        x2 = joint_coord_set[LIMB_CONNECT[limb_num][1], 0]
        y2 = joint_coord_set[LIMB_CONNECT[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 200 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 6),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), JOINT_COLOR_CODE[color_code_num]))

            cv2.fillConvexPoly(test_img, polygon, color=limb_color)

    return test_img


if __name__ == '__main__':
    tf.app.run()
