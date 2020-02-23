# -*- coding:utf-8 -*-
# @Time     :2019/3/6 15:03
# @Author   :zhuhejun


from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import random

def rotate_bound(img, angle):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    mat = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    mat[0, 2] += (nw / 2) - cx
    mat[1, 2] += (nh / 2) - cy

    return cv2.warpAffine(img, mat, (nw, nh), borderMode=cv2.BORDER_REPLICATE)

def gen_text(font_path, font_color, font_size, text, loc_status=1):
    font = ImageFont.truetype(font_path, font_size)

    height = int(font_size * 3)
    width = int(len(text) * height * 1.5)
    # width = int(font_size * 3)
    # height = int(len(text) * width * 1.5)
    if loc_status:
        ori_text_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))  # 255不透明 0全透明
        draw_handle = ImageDraw.Draw(ori_text_image, "RGBA")
        pos_list = []
        angle = np.random.random() * 2 - 1
        for char_idx, char in enumerate(text):
            # print(char )
            if char == ' ':
                continue
            tmp_text = text[:char_idx + 1]

            # draw_handle.text((3, 3), tmp_text, fill=font_color, font=font)
            draw_handle.text((3, 3), tmp_text, fill=font_color, font=font)
            text_image = cv2.cvtColor(np.asarray(ori_text_image), cv2.COLOR_RGBA2BGRA)
            text_image = rotate_bound(text_image, angle)
            deta_img = text_image[:, :, 3].copy()
            if len(pos_list):
                deta_img[:, :pos_list[-1][-1] + 2] = 0
                # deta_img[:pos_list[-1][-3] + 2, :] = 0
            pos = np.where(deta_img[:, :] > 100)
            # print('232',pos)
            # print('qqq',pos[0])
            # print('www',pos[1])
            pos_list.append([min(pos[0]), max(pos[0]), min(pos[1]), max(pos[1])])
            # print('qqq',pos_list)

        ori_text_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))  # 255不透明 0全透明

        draw_handle = ImageDraw.Draw(ori_text_image, "RGBA")
        # draw_handle.text((3, 3), text, fill=font_color, font=font)
        draw_handle.text((3, 3), text, fill=font_color, font=font)
        text_image = cv2.cvtColor(np.asarray(ori_text_image), cv2.COLOR_RGBA2BGRA)
        text_image = rotate_bound(text_image, angle)

        point_list = [[(x1 + x2) // 2, (y1 + y2) // 2] for y1, y2, x1, x2 in pos_list]
        deta_img = text_image[:, :, 3].copy()
        pos = np.where(deta_img[:, :] > 100)
        char_size = max(pos[0]) - min(pos[0])
        # char_size = max(pos[1]) - min(pos[1])
        # print(point_list)
        return text_image, char_size, point_list
    else:
        angle = np.random.random() * 2 - 1
        ori_text_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))  # 255不透明 0全透明

        draw_handle = ImageDraw.Draw(ori_text_image, "RGBA")

        draw_handle.text((3, 3), text, fill=font_color, font=font)

        # draw_handle.text((3, 3), '\n'.join(text), fill=font_color, font=font)
        text_image = cv2.cvtColor(np.asarray(ori_text_image), cv2.COLOR_RGBA2BGRA)
        text_image = rotate_bound(text_image, angle)

        # 字体变宽
        if np.random.random()>0.7:
            h,w = text_image.shape[:2]
            temp_size = random.uniform(1.1,2.1)
            size = (int(w*temp_size), h)
            text_image = cv2.resize(text_image,size)

        deta_img = text_image[:, :, 3].copy()
        pos = np.where(deta_img[:, :] > 100)
        # char_size = max(pos[1]) - min(pos[1])
        # 字高度
        char_size = max(pos[0])-min(pos[0])
        return text_image, char_size, None


if __name__ == '__main__':
    pass
