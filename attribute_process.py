import os
import numpy as np
import matplotlib.pyplot as plt

def load_parse_result(parse_file_path):
    img_attr_dict = {}
    with open(parse_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line_parts = line.strip().split(' ')
        img_path = line_parts[0]

        attr_list = []
        for attr in line_parts[1:]:
            if attr.endswith('ale'):
                if attr == 'Male':
                    attr_list.append(1.0)
                else:
                    attr_list.append(0.0)
                continue

            if attr.endswith(')'):
                attr = attr[:-6]

            try:
                attr_list.append(float(attr))
            except:
                continue

        img_attr_dict[img_path] = attr_list

    return img_attr_dict


def visualize_attribute_distribution(img_attr_dict):
    # transpose the parsing results
    attr_lists = []
    for k in img_attr_dict.keys():
        attr_lists.append(img_attr_dict[k])
    attr_lists = list(map(list, zip(*attr_lists)))

    # visualize the distribution of each attribute
    attr_names = ['gender', 'age', 'smile', 'pitch', 'roll', 'yaw',
                  'left_eye_no_glass_open', 'left_eye_no_glass_close',
                  'left_eye_normal_glass_open', 'left_eye_normal_glass_close',
                  'dark_glasses', 'occlusion',
                  'right_eye_no_glass_open', 'right_eye_no_glass_close',
                  'right_eye_normal_glass_open', 'right_eye_normal_glass_close',
                  'dark_glasses', 'occlusion',
                  'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise',
                  'surgical_mask_or_respirator', 'other_occlusion', 'close', 'open']

    for k, v in zip(attr_names, attr_lists):
        fig = plt.figure()
        plt.hist(v, density=True)
        plt.ylabel('Probability')
        plt.xlabel(k)
        plt.show()
        print(k)


def process_parse_file(img_attr_dict, attr_file_path):
    attr_list_total = []
    with open(attr_file_path, 'w') as f:
        for img_path in img_attr_dict.keys():
            attr_list_raw = img_attr_dict[img_path]
            attr_list = []
            
            # gender
            attr_list.append(str(attr_list_raw[0]))

            # age
            attr_list.append(str(attr_list_raw[1]))

            # smile
            attr_list.append(str(attr_list_raw[2]))

            # pose (profile vs. frontal)
            attr_list.append(str(attr_list_raw[5]))

            # wearing glasses
            no_glass_open = attr_list_raw[6] > 50.0
            no_glass_close = attr_list_raw[7] > 50.0
            attr_list.append('0.0' if (no_glass_open or no_glass_close) else '1.0')

            # happiness
            attr_list.append(str(attr_list_raw[21]))

            # mouth open
            attr_list.append(str(attr_list_raw[-1]))

            attr_list_total.append(attr_list)

            attr_list_str = ' '.join(attr_list)
            msg = img_path + ' ' + attr_list_str + '\n'
            f.write(msg)
            
    return attr_list_total


if __name__ == '__main__':
    batch_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for batch_ind in batch_inds:
        parse_file_path = './sample/api_res_raw/img_attr_raw_{}.txt'.format(batch_ind)
        img_attr_dict = load_parse_result(parse_file_path)
        
        # visualize_attribute_distribution(img_attr_dict)

        attr_file_path = './sample/api_res_fine/img_attr_{}.txt'.format(batch_ind)
        process_parse_file(img_attr_dict, attr_file_path)
    