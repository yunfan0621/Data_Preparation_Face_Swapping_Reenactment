import os
import time
import argparse
import base64
from tqdm import tqdm

import cv2
from PythonSDK.facepp import API, File

api = API()

# api interface for attribute estimation
def att_estimation_api(img_path, args):
    global api

    cnt = 0
    time_out = 10

    # handle invalid image paths
    if not os.path.exists(img_path):
        print('Invalid image path! (%s)' % (img_path))
        return -1

    while True:
        cnt += 1
        time.sleep(0.5)
        try:
            # read in the image and resize
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb_resize = cv2.resize(img_rgb, dsize=(args.load_size, args.load_size))

            # convert to base64 data
            img_str = cv2.imencode('.jpg', img_rgb_resize)[1].tostring()
            img_base64 = base64.b64encode(img_str)
            det_res = api.detect(image_base64=img_base64, return_attributes="smiling,headpose,eyestatus,emotion,mouthstatus")
            # det_res = api.detect(image_file=File(img_path), return_attributes="gender,age,smiling,headpose,"
            #                                                                  "eyestatus,emotion,mouthstatus")
            if 'faces' in det_res and det_res['faces']:
                return det_res
            else:
                return -1
        except:
            if cnt >= time_out:
                print('Time out!')
                return -1


def parse_api_result(det_res):
    res_dict = det_res['faces'][0]['attributes']

    msg = ''
    #msg += res_dict['gender']['value'] + ' '
    #msg += str(res_dict['age']['value']) + ' '
    msg += '{}({})'.format(str(res_dict['smile']['value']), str(res_dict['smile']['threshold'])) + ' '

    msg += '{} {} {}'.format(str(res_dict['headpose']['pitch_angle']),
                             str(res_dict['headpose']['roll_angle']),
                             str(res_dict['headpose']['yaw_angle'])) + ' '

    msg += '{} {} {} {} {} {}'.format(str(res_dict['eyestatus']['left_eye_status']['no_glass_eye_open']),
                                      str(res_dict['eyestatus']['left_eye_status']['no_glass_eye_close']),
                                      str(res_dict['eyestatus']['left_eye_status']['normal_glass_eye_open']),
                                      str(res_dict['eyestatus']['left_eye_status']['normal_glass_eye_close']),
                                      str(res_dict['eyestatus']['left_eye_status']['dark_glasses']),
                                      str(res_dict['eyestatus']['left_eye_status']['occlusion'])) + ' '

    msg += '{} {} {} {} {} {}'.format(str(res_dict['eyestatus']['right_eye_status']['no_glass_eye_open']),
                                      str(res_dict['eyestatus']['right_eye_status']['no_glass_eye_close']),
                                      str(res_dict['eyestatus']['right_eye_status']['normal_glass_eye_open']),
                                      str(res_dict['eyestatus']['right_eye_status']['normal_glass_eye_close']),
                                      str(res_dict['eyestatus']['right_eye_status']['dark_glasses']),
                                      str(res_dict['eyestatus']['right_eye_status']['occlusion'])) + ' '

    msg += '{} {} {} {} {} {} {}'.format(str(res_dict['emotion']['anger']),
                                         str(res_dict['emotion']['disgust']),
                                         str(res_dict['emotion']['fear']),
                                         str(res_dict['emotion']['happiness']),
                                         str(res_dict['emotion']['neutral']),
                                         str(res_dict['emotion']['sadness']),
                                         str(res_dict['emotion']['surprise'])) + ' '

    msg += '{} {} {} {}'.format(str(res_dict['mouthstatus']['surgical_mask_or_respirator']),
                                str(res_dict['mouthstatus']['other_occlusion']),
                                str(res_dict['mouthstatus']['close']),
                                str(res_dict['mouthstatus']['open'])) + ' '

    return msg


def load_parse_result(parse_file_path):
    img_attr_dict = {}
    with open(parse_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line_parts = line.strip('\n').split(' ')
        img_path = line_parts[0]
        attr = ' '.join(line_parts[1:])
        img_attr_dict[img_path] = attr

    return img_attr_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_size', type=int, default=256, help='image size for the input to API')
    parser.add_argument('--batch_size', type=int, default=10000, help='number of images in a single process')
    parser.add_argument('--batch_ind', type=int, default=0, help='index of the image batch')

    args = parser.parse_args()

    args.img_dir = "/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset/image_{}".format(args.load_size)

    # load the parse results for resuming testing (if exists)
    parse_file_path = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset/attr_label_{}/raw_detections/img_attr_raw_{}.txt'.format(args.load_size, args.batch_ind)
    if os.path.exists(parse_file_path):
        img_attr_dict = load_parse_result(parse_file_path)
    else:
        img_attr_dict = {}

    # go through the images and do API test (skip if done)
    img_list = os.listdir(args.img_dir)[args.batch_ind*args.batch_size : (args.batch_ind+1)*args.batch_size]
    for img_ind in tqdm(range(len(img_list))):
        img_name = img_list[img_ind]
        img_path = os.path.join(args.img_dir, img_name)

        # skip if already tested
        if img_path in img_attr_dict.keys():
            continue

        res = att_estimation_api(img_path, args)
        if res == -1:
            print('API analysis fails for {}'.format(img_path))
        else:
            img_attr_dict[img_path] = parse_api_result(res)

    # dump test results to file
    with open(parse_file_path, 'w') as f:
        for k in img_attr_dict.keys():
            msg = k + ' ' + img_attr_dict[k] + '\n'
            f.write(msg)