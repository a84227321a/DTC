import os
import re
import numpy as np
import random
import cv2
from pre_util import *
import keras.backend as K
import itertools
import keras.callbacks

OUTPUT_DIR =''

def gen_text(char_list, count=(10, 15)):
    chars = [np.random.choice(char_list) for _ in range(np.random.randint(count[0], count[1]))]
    text = ''.join(chars)
    return text

def min_ctc_len(text):
    ctc_len = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            ctc_len += 2
        else:
            ctc_len += 1
    return ctc_len

def load_bg_images(root):
    bg_image_names = list()
    names = os.listdir(root)
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            bg_image_names.extend(load_bg_images(path))
    return bg_image_names

def gen_train_img(text, target_h,bg_image_path):
    bg_image_path_list = load_bg_images(bg_image_path)
    bg_image_path = np.random.choice(bg_image_path_list)
    bg_image = cv2.imread(bg_image_path)
    bg_h, bg_w = bg_image.shape[:2]
    # tmp_h = np.random.randint(32, 50)
    tmp_h = np.random.randint(32, int(bg_w/len(text)+1))
    tmp_w = np.random.randint((len(text) - 2) * tmp_h, len(text) * tmp_h)
    x_offset = np.random.randint(bg_w - tmp_w)
    y_offset = np.random.randint(bg_h - tmp_h)
    bg_image = bg_image[y_offset:y_offset + tmp_h, x_offset: x_offset + tmp_w]
    image,points = imGenerator.gen(bg_image, text, {'loc': 0})
    h, w = image.shape[:2]


    if w / h > len(text) * 1.5 or w * 8 / h < min_ctc_len(text) + 1:
        return None
    rate = target_h / h
    img_w = int(w * rate)
    image = cv2.resize(image, (img_w, target_h))
    image = image.astype(np.float)
    image = image / 255.
    if len(text) > ((img_w + 1) // 2 + 1) // 2 // 2:
        return None
    return image, img_w, points

def get_train(img_path, txt_path, target_h):
    pathDir = os.listdir(img_path)
    im_name = random.sample(pathDir,1)[0]
    im_names = os.path.join(img_path,im_name)
    image = cv2.imread(im_names)
    image = rotate_bound(image, angle=np.random.random() * 2 - 1)

    h, w = image.shape[:2]

    target_w = int(w * target_h / h * (0.9 + np.random.random() * 0.2))
    image = cv2.resize(image, (target_w, target_h))

    noise_func = np.random.choice([speckle, gaussian_noise, pepperand_salt_noise,
                                   poisson_noise, gamma_noise, brightened, colored,contrasted,sharped])
    noise_type = np.random.choice([0, 1])
    # 增加噪声
    image = noise_func(image, noise_type=noise_type)

    # img = gray_stretch(img)
    image = image.astype(np.float)
    image = image / 255.0
    tmp_list = im_name.split('.')
    del tmp_list[-1]
    tmp_list.append('txt')
    txt_name = ('.'.join(tmp_list))
    txt_names = os.path.join(txt_path,txt_name)
    with open(txt_names, 'rb', encoding='utf-8') as txt_file:
        txt = txt_file.read().splitlines()[0]
    return image,target_w,txt

def load_img_test(img_path, target_h):
    img_path = re.sub('\\\\', '/', img_path)
    img = cv2.imread(img_path)
    # add
    h, w = img.shape[:2]
    target_w = int(w * target_h / h)
    # print(img_path)
    img = cv2.resize(img, (target_w, target_h))
    # img = gray_stretch(img)
    img = img.astype(np.float)
    img = img / 255.0
    # x = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
    # x = np.array(x)
    return img

def edit_distance(word1, word2):
    len1 = len(word1);
    len2 = len(word2);
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i;
    for j in range(len2 + 1):
        dp[0][j] = j;

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 2
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


class TextImageGenerator(keras.callbacks.Callback):
    def __init__(self, train_dir, char_idx_dict,batch_size,img_h,max_string_len,bg_image_path):
        super().__init__()
        self.batch_size = batch_size
        self.img_h = img_h
        self.char_idx_dict = char_idx_dict
        self.chars_list = list(self.char_idx_dict.keys())
        self.blank_label = self.output_size() - 1
        self.max_string_len = max_string_len
        self.img_tr_path = os.path.join(train_dir,'imgs')
        self.txt_tr_path = os.path.join(train_dir,'txts')
        self.bg_image_path = bg_image_path
        self.train_scale = 0.5
    def output_size(self):
        return len(self.char_idx_dict) + 1

    def get_batch(self, size):
        imgs = list()
        img_w_list = list()
        labels = np.ones([size, self.absolute_max_string_len]) * self.blank_label
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        for i in range(size):
            while 1:
                try:
                    if np.random.random()>self.train_scale :
                        img, real_img_w, text = get_train(self.img_tr_path, self.txt_tr_path, target_h=self.img_h)
                        if real_img_w > self.img_h * 13 and len(text) > ((real_img_w + 1) // 2 + 1) // 2 // 2:
                            continue
                        break
                    else:
                        text = gen_text(self.chars_list, (10, 11))
                        gen_result = gen_train_img(text, target_h=self.img_h,bg_image_path=self.bg_image_path)
                        img, real_img_w, _points = gen_result
                        break
                except Exception as e:
                    pass
            imgs.append(img)
            img_w_list.append(real_img_w)
            label = list()
            text = text.replace(' ', '')
            for char in text:
                label.append(self.char_idx_dict[char])
            label_len = len(label)
            labels[i, :label_len] = label
            input_length[i] = ((real_img_w + 1) // 2 + 1) // 2
            label_length[i] = label_len

        img_w = np.max(img_w_list)

        if K.image_data_format() == 'channels_first':
            x_data = np.ones([size, 3, img_w, self.img_h])
        else:
            x_data = np.ones([size, self.img_h, img_w, 3])
        for img_idx in range(len(img_w_list)):
            if K.image_data_format() == 'channels_first':
                x_data[img_idx, :, :img_w_list[img_idx], :] = imgs[img_idx]
            else:
                x_data[img_idx, :, :img_w_list[img_idx], :] = imgs[img_idx]

        inputs = {'the_input': x_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([size])}
        return inputs, outputs
    def next_train(self):
        while True:
            ret = self.get_batch(self.batch_size)
            yield ret

class ValGenerator(keras.callbacks.Callback):
    def __init__(self,save_dir,batch_size,prediction_model,img_h,sample_list,idx_char_dict):
        super().__init__()
        self.output_dir = os.path.join(
            OUTPUT_DIR, save_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.base_model = prediction_model
        self.img_h = img_h

        self.idx_char_dict = idx_char_dict
        self.sample_list = sample_list
        self.label_len = len(sample_list)

        self.batch_size = batch_size
        self.img_name_list = []
        self.label_list = []
        self.init_sample()
        self.paint_func = lambda text: load_img_test(text, self.img_h)
        self.max_acc = -1
    def init_sample(self):
        for img_name, label in self.sample_list:
            self.img_name_list.append(img_name)
            self.label_list.append(label)

    def get_batch(self, index, size):
        x_data = []
        labels = []
        for i in range(size):
            if K.image_data_format() == 'channels_first':
                x_data.append(self.paint_func(self.img_name_list[index + i]))
            else:
                x_data.append(self.paint_func(self.img_name_list[index + i]))
            labels.append(self.label_list[index + i])

        x_data = np.array(x_data)
        return x_data, labels

    def cal_acc(self, y_true, y_pred, tmp_img_name_list):
        correct_count = 0
        y = np.argmax(y_pred, axis=-1)
        y_pred = []
        for i in range(len(y)):
            text = ''
            for t_y_j in [key for key, _k in itertools.groupby(y[i])]:
                if t_y_j in self.idx_char_dict:
                    text += self.idx_char_dict[t_y_j]
            y_pred.append(text)

        for i in range(len(y_true)):
            y_true_i = y_true[i]
            y_pred_i = y_pred[i]
            if y_true_i == y_pred_i:
                correct_count += 1
            else:
                print(tmp_img_name_list[i], y_true_i, y_pred_i)

        return correct_count,y_true_i,y_pred_i

    def on_epoch_end(self, epoch, logs=None):
        cur_test_index = 0
        correct_count = 0
        edit_dis = 0

        av_distance = 0
        len_all_str = 0
        while 1:
            if cur_test_index + self.batch_size >= self.sample_len:
                x_data, y_true = self.get_batch(cur_test_index, self.sample_len - cur_test_index)
                y_pred = self.base_model.predict(x_data)
                cal, y_true_t, y_pred_t = self.cal_acc(y_true, y_pred,
                                                       self.img_name_list[cur_test_index:self.sample_len])
                correct_count += cal

                av_distance += edit_distance(y_true_t, y_pred_t)
                temp_len = len(y_true_t) + len(y_pred_t)
                len_all_str += temp_len
                distance = int(edit_distance(y_true_t, y_pred_t)) / (len(y_true_t) + len(y_pred_t))
                edit_dis += distance
                break
            else:
                x_data, y_true = self.get_batch(cur_test_index, self.batch_size)
                y_pred = self.base_model.predict(x_data)
                cal, y_true_t, y_pred_t = self.cal_acc(y_true, y_pred,
                                                       self.img_name_list[
                                                       cur_test_index:cur_test_index + self.batch_size])
                correct_count += cal
                cur_test_index += self.batch_size

                av_distance += edit_distance(y_true_t, y_pred_t)
                temp_len = len(y_true_t) + len(y_pred_t)

                len_all_str += temp_len
                distance = int(edit_distance(y_true_t, y_pred_t)) / (len(y_true_t) + len(y_pred_t))
                edit_dis += distance
        acc = correct_count / self.sample_len
        av_edit_distance = edit_dis / self.sample_len
        av_distance = int(av_distance)
        an_av_edit = av_distance/len_all_str
        print('acc:{}'.format(acc))
        print('av_edit_distance:{}'.format(av_edit_distance))
        print('an_av_edit_distance:{}'.format(an_av_edit))
        self.model.save_weights(os.path.join(self.output_dir, 'weights.h5'))
        self.model.save(os.path.join(self.output_dir, 'model.h5'))
        if acc >= self.max_acc:
            self.base_model.save_weights(os.path.join(self.output_dir, 'base_weights.h5'))
            self.base_model.save(os.path.join(self.output_dir, 'base_model.h5'))
            self.max_acc = acc
            print('{} epoch model change'.format(epoch))