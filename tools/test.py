import os 
import cv2
import numpy as np
import csv

# mot_dir = "/home/max/Downloads/MOT16/train/"    
    
    
# for sequence in os.listdir(mot_dir):
#     print("Processing %s" % sequence)
#     sequence_dir = os.path.join(mot_dir, sequence)

#     image_dir = os.path.join(sequence_dir, "img1")
#     image_filenames = {
#         int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
#         for f in os.listdir(image_dir)}
#     print(image_filenames)


# yt_dir = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/Test_dataset/"
# bbox_dir = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_csv_bbox/"

# files = []
# for f in sorted(os.listdir(bbox_dir)):
#     domain = os.path.abspath(bbox_dir)
#     f = os.path.join(domain,f)
#     files += [f]
#     for line in open(f, "r"):
#         data = line.split(",")
#         filename = data[0]
#         im_filename = os.path.join(yt_dir,filename)
#         print(im_filename)
    

# for filename_txt in os.listdir(bbox_dir):
#     detection_file = os.path.join(bbox_dir, filename_txt)
#     #print(detection_file)




# def change_to_mot(input_dir, middle_dir):
#     array_dic = []
#     for line in open(input_dir, "r"):
#         data = line.split(",")
#         total_len = len(data)
#         filename = data[0]
#         label = data[1]
#         new_label = label.replace('face', '-1')
#         bbox_info = [float(i) for i in data[2:total_len]]
#         xmin = bbox_info[0]
#         xmax = bbox_info[1]
#         ymin = bbox_info[2]
#         ymax = bbox_info[3]
#         array_dic.append(np.array([filename, new_label, xmin, xmax, ymin, ymax]))
#     a = np.array(array_dic)
#     np.savetxt(middle_dir, a, fmt="%s,%s,%s,%s,%s,%s")
#     print(middle_dir)

# def add_last_four(middle_dir, output_dir):
#     with open(middle_dir, 'r') as f:
#         data = csv.reader(f, delimiter=',')

#         with open(output_dir, 'w') as f:
#             for r in data:
#                 f.write('{},{},{},{},{},{},1,-1,-1,-1\n'.format(*r))                
            
# def main():
#     input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_csv_bbox/'
#     middle_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_info_csv/'
#     output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_output_csv/'
    
#     if not os.path.exists(middle_path):
#         os.mkdir(middle_path)
#     for filename in os.listdir(input_path):
#         if os.path.isfile(os.path.join(input_path,filename)):
#             change_to_mot(input_path+filename, middle_path+filename)
    
#     if not os.path.exists(output_path):
#         os.mkdir(output_path)
#     for filename in os.listdir(middle_path):
#         if os.path.isfile(os.path.join(middle_path,filename)):
#             add_last_fours(middle_path+filename, output_path+filename)    


# if __name__ == "__main__":
#     main()


# def delete_extra(input_dir, output_dir):
#     array_dic = []
#     for line in open(input_dir, 'r'):
#         data = line.split(",")
#         data[-1] = -1
#         filename_ext = data[0]
#         filename = os.path.splitext(filename_ext)[0]
#         filename_data = filename.split("/")
#         filename_id = filename_data[2]
#         total_len = len(data)
#         #print(total_len)
#         #print(filename_id)
#         array_dic.append(np.array([filename_id, data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]]))   
#     a = np.array(array_dic)
#     print(a) 
#     np.savetxt(output_dir, a, fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s,%s")
#     #print(output_dir)
        

# def main():
#     input_path = '/home/maxwell/Downloads/deep_sort/dataset/test_csv/'
#     output_path = '/home/maxwell/Downloads/deep_sort/dataset/test_output_csv/'

#     if not os.path.exists(output_path):
#         os.mkdir(output_path)
#     for filename in os.listdir(input_path):
#         if os.path.isfile(os.path.join(input_path, filename)):
#             delete_extra(input_path+filename, output_path+filename)

# if __name__ == "__main__":
#     main()



# csv_dir = '/home/max/Downloads/cosine_metric_learning/datasets/test_csv'
# for filename in os.listdir(csv_dir):
#     for line in open(os.path.join(csv_dir,filename), "r"):
#         data = line.split(",")
#         filename = data[0]
#         filename_base, ext = os.path.splitext(filename)
#         person_name, dir_file, frame_idx = filename_base.split("/")
#         print(person_name)
#         print(dir_file)
#         print(frame_idx)
 

# PATH_TO_TEST_IMAGES_DIR = "/home/max/Downloads/MTCNN/models/research/object_detection/samples/data/lfpw_testImage/"
# #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
# for image_file in sorted(os.listdir(PATH_TO_TEST_IMAGES_DIR)):
#     images = os.path.join(PATH_TO_TEST_IMAGES_DIR, image_file)
#     print(images)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from skimage import io, transform, color, util

flags = tf.flags
flags.DEFINE_string(flag_name='directory', default_value='/home/a/Datasets/cat&dog/class', docstring='数据地址')
flags.DEFINE_string(flag_name='save_dir', default_value='./tfrecords', docstring='保存地址')
flags.DEFINE_integer(flag_name='test_size', default_value=350, docstring='测试集大小')
FLAGS = flags.FLAGS

MODES = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_tfrecord(mode, anno):
    """转换为TfRecord"""

    assert mode in MODES, "模式错误"

    filename = os.path.join(FLAGS.save_dir, mode + '.tfrecords')

    with tf.python_io.TFRecordWriter(filename) as writer:
        for fnm, cls in tqdm(anno):

            # 读取图片、转换
            img = io.imread(fnm)
            img = color.rgb2gray(img)
            img = transform.resize(img, [224, 224])

            # 获取转换后的信息
            if 3 == img.ndim:
                rows, cols, depth = img.shape
            else:
                rows, cols = img.shape
                depth = 1

            # 创建Example对象
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/height': _int_feature(rows),
                        'image/width': _int_feature(cols),
                        'image/depth': _int_feature(depth),
                        'image/class/label': _int_feature(cls),
                        'image/encoded': _bytes_feature(img.astype(np.float32).tobytes())
                    }
                )
            )
            # 序列化并保存
            writer.write(example.SerializeToString())


def get_folder_name(folder):
    """不递归，获取特定文件夹下所有文件夹名"""

    fs = os.listdir(folder)
    fs = [x for x in fs if os.path.isdir(os.path.join(folder, x))]
    return sorted(fs)


def get_file_name(folder):
    """不递归，获取特定文件夹下所有文件名"""

    fs = os.listdir(folder)
    fs = map(lambda x: os.path.join(folder, x), fs)
    fs = [x for x in fs if os.path.isfile(x)]
    return fs


def get_annotations(directory, classes):
    """获取所有图片路径和标签"""

    files = []
    labels = []

    for ith, val in enumerate(classes):
        fi = get_file_name(os.path.join(directory, val))
        files.extend(fi)
        labels.extend([ith] * len(fi))

    assert len(files) == len(labels), "图片和标签数量不等"

    # 将图片路径和标签拼合在一起
    annotation = [x for x in zip(files, labels)]

    # 随机打乱
    random.shuffle(annotation)

    return annotation


def main(_):
    class_names = get_folder_name(FLAGS.directory)
    annotation = get_annotations(FLAGS.directory, class_names)

    convert_to_tfrecord(tf.estimator.ModeKeys.TRAIN, annotation[FLAGS.test_size:])
    convert_to_tfrecord(tf.estimator.ModeKeys.EVAL, annotation[:FLAGS.test_size])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
