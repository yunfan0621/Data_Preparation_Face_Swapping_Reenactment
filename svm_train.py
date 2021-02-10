import os 
import numpy as np
import ntpath
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
from model import Generator
from utils import latent_list2tensor

from pdb import set_trace as ST


def train_svm_boundary(latent_codes, scores, split_ratio=0.7, regularizer='l2', space_name='z', chosen_num_or_ratio=0.05):
    n_samples, n_features = latent_codes.shape

    ''' sort the data according to scores (in descending order) '''
    sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
    latent_codes = latent_codes[sorted_idx]
    scores = scores[sorted_idx]

    ''' deal with the number of samples for positive / negative class '''
    if np.max(scores) == 1.0 and np.min(scores) == -1.0:
        n_pos = np.count_nonzero(scores >= 0.8)
        n_neg = np.count_nonzero(scores <= -0.8)
        n_chosen = min(n_pos, n_neg) # choose whichever class with fewer labels with high confidence
    else:
        if 0 < chosen_num_or_ratio <= 1:
           n_chosen = int(n_samples * chosen_num_or_ratio)
        else:
           n_chosen = int(chosen_num_or_ratio)

    n_chosen = min(n_chosen, n_samples // 2)

    ''' make train / val split and select pos / neg samples '''
    n_train = int(n_chosen * split_ratio)
    n_val = n_chosen - n_train

    # positive samples (select from the beginning of data)
    pos_idx = np.arange(n_chosen)
    np.random.shuffle(pos_idx)
    pos_train = latent_codes[:n_chosen][pos_idx[:n_train]]
    pos_val = latent_codes[:n_chosen][pos_idx[n_train:]]
    # pos_train_scores = scores[:n_chosen][pos_idx[:n_train]]
    # pos_val_scores = scores[:n_chosen][pos_idx[n_train:]]

    # negative samples (select from the end of data)
    neg_idx = np.arange(n_chosen)
    np.random.shuffle(neg_idx)
    neg_train = latent_codes[-n_chosen:][neg_idx[:n_train]]
    neg_val = latent_codes[-n_chosen:][neg_idx[n_train:]]
    # neg_train_scores = scores[-n_chosen:][neg_idx[:n_train]]
    # neg_val_scores = scores[-n_chosen:][neg_idx[n_train:]]

    # Training set.
    train_data = np.concatenate([pos_train, neg_train], axis=0)
    train_label = np.concatenate([np.ones(n_train, dtype=np.int), np.zeros(n_train, dtype=np.int)], axis=0)

    # Validation set.
    val_data = np.concatenate([pos_val, neg_val], axis=0)
    val_label = np.concatenate([np.ones(n_val, dtype=np.int), np.zeros(n_val, dtype=np.int)], axis=0)

    # Remaining set.
    n_remaining = n_samples - n_chosen * 2
    remaining_data = latent_codes[n_chosen : -n_chosen]
    remaining_scores = scores[n_chosen : -n_chosen]
    decision_value = (scores[0] + scores[-1]) / 2
    remaining_label = np.ones(n_remaining, dtype=np.int)
    remaining_label[remaining_scores.ravel() < decision_value] = 0


    ''' training '''
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(penalty=regularizer,
                                  dual=(n_samples<=n_features),
                                  tol=1e-4,
                                  verbose=False,
                                  max_iter=10000))
    classifier = clf.fit(train_data, train_label)


    ''' validating and testing '''
    msg = ''

    train_prediction = classifier.predict(train_data)
    train_correct_num = np.sum(train_label == train_prediction)
    train_acc = train_correct_num * 1.0 / (n_train * 2)
    train_msg = f'Accuracy for training set: ' + \
                f'{train_correct_num} / {n_train * 2} = ' + \
                f'{train_correct_num / (n_train * 2):.6f}'
    print(train_msg)
    msg = msg + train_msg + '\n'

    val_prediction = classifier.predict(val_data)
    val_correct_num = np.sum(val_label == val_prediction)
    val_acc = val_correct_num * 1.0 / (n_val * 2)
    val_msg = f'Accuracy for validate set: ' + \
              f'{val_correct_num} / {n_val * 2} = ' + \
              f'{val_correct_num / (n_val * 2):.6f}'
    print(val_msg)
    msg = msg + val_msg + '\n'

    remaining_prediction = classifier.predict(remaining_data)
    remaining_correct_num = np.sum(remaining_label == remaining_prediction)
    remaining_acc = remaining_correct_num * 1.0 / n_remaining
    remaining_msg = f'Accuracy for remaining set: ' + \
                    f'{remaining_correct_num} / {n_remaining} = ' + \
                    f'{remaining_correct_num / n_remaining:.6f}'
    print(remaining_msg)
    msg = msg + remaining_msg + '\n'

    total_correct = train_correct_num + val_correct_num + remaining_correct_num
    total_acc = total_correct * 1.0 / n_samples
    total_msg = f'Accuracy for the entire set: ' + \
                f'{total_correct} / {n_samples} = ' + \
                f'{total_correct / n_samples:.6f}'
    print(total_msg)
    msg = msg + total_msg + '\n'

    ''' analyze stat of the normal vector '''
    w = classifier[1].coef_.reshape(1, n_features).astype(np.float32)
    b = classifier[1].intercept_.astype(np.float32)
    
    w_nonzero = np.count_nonzero(w)
    w_all = w.size
    w_msg = f'Sparcity of the normal vector: ' + \
            f'{w_nonzero} / {w_all}'
    print(w_msg)
    msg = msg + w_msg + '\n'

    return w, b, msg


def load_img_attr_file(parse_file_path):
    img_attr_dict = {}

    if isinstance(parse_file_path, list):
        for p in parse_file_path:
            tmp_img_attr_dict = load_img_attr_file(p)
            for k in tmp_img_attr_dict.keys():
                img_attr_dict[k] = tmp_img_attr_dict[k]

        return img_attr_dict

    with open(parse_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line_parts = line.strip().split(' ')
        img_path = line_parts[0]

        attr_list = []
        for attr in line_parts[1:]:
            attr_list.append(float(attr))

        img_attr_dict[img_path] = attr_list

    return img_attr_dict


def load_data_label(img_attr_dict, space_name, attr_ind, latent_dir):
    n_samples = len(img_attr_dict.keys())
    if '256' in latent_dir:
        n_features = 8704 if space_name == 'style' else 512
    else:
        n_features = 9088 if space_name == 'style' else 512
    # n_attributes = len(img_attr_dict[list(img_attr_dict.keys())[0]])

    latent_code_data_mat = np.zeros([n_samples, n_features])
    # latent_code_label_mat = np.zeros([n_samples, n_attributes])
    latent_code_label_mat = np.zeros([n_samples, 1]) # load for one attribute at a time

    n_imgs = len(img_attr_dict.keys())
    for img_ind in tqdm(range(n_imgs)):
        img_path = list(img_attr_dict.keys())[img_ind]

        # load data
        latent_code_name = ntpath.basename(img_path)[:-4] + '.pt'
        latent_code_path = os.path.join(latent_dir, latent_code_name)
        latent_code_data = torch.load(latent_code_path)[space_name]
        latent_code_data_mat[img_ind, :] = latent_list2tensor(latent_code_data) if space_name == 'style' else latent_code_data.cpu().numpy()

        # load label
        latent_code_label = img_attr_dict[img_path][attr_ind:attr_ind+1] # keep the dimension
        latent_code_label_mat[img_ind, :] = np.array([latent_code_label])

    return latent_code_data_mat, latent_code_label_mat


if __name__ == '__main__':
    img_size = 256
    split_ratio = 0.7
    
    # load annotations obtained by API
    parse_file_ind  = [0, 1, 2, 3, 4]
    parse_file_path = ['/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset/attr_label_{}/refined_detections/img_attr_fine_{}.txt'.format(img_size, i) for i in parse_file_ind]
    img_attr_dict = load_img_attr_file(parse_file_path)

    attr_name_list  = ['smile', 'pitch', 'roll', 'yaw', 'eyes_open', 'happiness', 'mouth_open']
    space_name_list = ['style', 'w',     'w',    'w',   'style',     'style',     'style']

    # load latent codes and correpsonding labels according to the space name
    for attr_ind, (attr_name, space_name) in enumerate(zip(attr_name_list, space_name_list)):
        regularizer = 'l1' if space_name == 'style' else 'l2'

        with open('/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset/log_{}/svm_log_{}_{}_{}.txt'\
                 .format(img_size, attr_name, regularizer, space_name), 'w') as f:
            print('Training SVM boundary for \'{}\' ({})'.format(attr_name, space_name))
            
            # load data
            latent_dir = '/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset/latent_{}'.format(img_size)
            train_data, train_label = load_data_label(img_attr_dict, space_name, attr_ind, latent_dir) # load one attribute at a time
            w, b, msg = train_svm_boundary(train_data, train_label, split_ratio=split_ratio, 
                                           regularizer=regularizer, space_name=space_name, chosen_num_or_ratio=0.05)

            # log the result
            f.write('Attribute = {}\n'.format(attr_name))
            # f.write('Ratio = {}\n'.format(chosen_ratio))
            f.write(msg + '\n')

            # save the boundary normal
            save_dir = os.path.join('/data/yunfan.liu/Data_Preparation_Face_Swapping_Reenactment/sample/svm_trainset/svm_normals_256', space_name, regularizer)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'n_{}.pt'.format(attr_name))
            torch.save({'w' : w, 'b' : b}, save_path)