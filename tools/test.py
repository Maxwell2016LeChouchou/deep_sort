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



csv_dir = '/home/maxwell/Downloads/cosine_metric_learning/datasets/test_csv'
for filename in os.listdir(csv_dir):
    for line in open(os.path.join(csv_dir,filename), "r"):
        data = line.split(",")
        filename = data[0]
        filename_base, ext = os.path.splitext(filename)
        person_name, dir_file, frame_idx = filename_base.split("/")
    print(filename)
    # print(person_name)
    # print(dir_file)
    # print(frame_idx)

# image_path = '/home/maxwell/Downloads/cosine_metric_learning/datasets/test_dataset'

# image_list = []
# yt_person_name = []
# yt_dir_list = []

# for home, dirs, files in os.walk(image_path):
#     for filename in files:
#         meta_data =