__author__ = 'Wang07'
import numpy as np
import cv2
import os
from font_util import FontHelper
from gen_text_util import gen_text
from pre_util import *


class ImageGenerator:
    def __init__(self, font_dir):
        self.__font_helper = FontHelper(font_dir)

    def gen(self, bg_image, text, options):
        font_color = np.random.randint(0, 20, 3)
        font_color = (font_color[0], font_color[1], font_color[2])
        font_paths = self.__font_helper.get_fonts(text)
        font_path = font_paths[np.random.randint(len(font_paths))]
        font_size = np.random.randint(25, 32)
        if type(options) is dict and 'loc' in options:
            text_image, char_size, point_list = gen_text(font_path=font_path, font_color=font_color,
                                                         font_size=font_size, text=text, loc_status=options['loc'])

        else:
            text_image, char_size, point_list = gen_text(font_path=font_path, font_color=font_color,
                                                         font_size=font_size, text=text)
        text_image, point_list = crop_image(text_image, point_list)
        if np.random.random() > 0.2:
            bg_image = dotted_line(bg_image)
        text_image = add_background(text_image, bg_image)
        if point_list is None:
            point_list = []
        else:
            point_list = [[int(x), int(y)] for x, y in point_list]
        noise_func = np.random.choice([speckle, gaussian_noise, pepperand_salt_noise,poisson_noise, gamma_noise])
        noise_type = np.random.choice([0, 1])
        if np.random.random() > 0.5:
            text_image = noise_func(text_image, noise_type=noise_type)
            text_image = smooth(text_image, char_size=char_size)
        else:
            text_image = smooth(text_image, char_size=char_size)
            text_image = noise_func(text_image, noise_type=noise_type)
        enhance_func = np.random.choice([brightened, colored, contrasted, sharped])
        text_image = enhance_func(text_image)
        return text_image, point_list


def min_ctc_len(text):
    ctc_len = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            ctc_len += 2
        else:
            ctc_len += 1
    return ctc_len


if __name__ == '__main__':
    bg_image_path_x = r'F:\ocr\bg_image'
    imGenerator = ImageGenerator(r'F:\ocr\字体')
    def load_bg_images(root):
        bg_image_names = list()
        names = os.listdir(root)
        for name in names:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                bg_image_names.extend(load_bg_images(path))
            if os.path.isfile(path) and (name.endswith('jpg') or name.endswith('png')):
                bg_image_names.append(path)

        return bg_image_names

    def gen_train_img(text, target_h):
        bg_image_path_list = load_bg_images(bg_image_path_x)
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
        image = image / 255.
        if len(text) > ((img_w + 1) // 2 + 1) // 2 // 2:
            return None
        return image, img_w, points

    def gen_text1(char_list, count=(10, 15)):
        # chars = list(np.random.choice(char_list, np.random.randint(count)))
        chars = [np.random.choice(char_list) for _ in range(np.random.randint(count[0], count[1]))]
        text = ''.join(chars)
        return text

    import random
    import os

    cout = 0
    while 1:
        text = '央视著名节目主持人'
        # temp_text = ''.join(text)
        gen_result = gen_train_img(text, target_h=25)
        try:
            img, real_img_w, _points = gen_result
        except:
            print('error')
            continue
        print(img.shape)
        img = img * 255.
        img = img.astype(np.uint8)
        cv2.imshow('111', img)
        cv2.waitKey(0)
