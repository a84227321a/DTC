import random
from PIL import Image
from PIL import ImageEnhance
import cv2
import numpy as np
from scipy import ndimage

def brightened(img):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    enh_bri = ImageEnhance.Brightness(img)
    brightness = random.uniform(0.5,1.5)
    image_brightened = enh_bri.enhance(brightness)
    image_brightened = cv2.cvtColor(np.asarray(image_brightened),cv2.COLOR_RGB2BGR)
    return image_brightened

def colored(img):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    enh_col = ImageEnhance.Color(img)
    color = random.uniform(0.5,1.5)
    image_colored = enh_col.enhance(color)
    image_colored = cv2.cvtColor(np.asarray(image_colored),cv2.COLOR_RGB2BGR)
    return image_colored

def contrasted(img):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    enh_con = ImageEnhance.Contrast(img)
    contrast = random.uniform(0.5,1.5)
    image_constasted = enh_con.enhance(contrast)
    image_constasted = cv2.cvtColor(np.asarray(image_constasted),cv2.COLOR_RGB2BGR)
    return image_constasted

def sharped(img):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    enh_sha = ImageEnhance.Sharpness(img)
    sharpness = random.uniform(0.5,2.5)
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped = cv2.cvtColor(np.asarray(image_sharped),cv2.COLOR_RGB2BGR)
    return image_sharped

def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
# np.random.gamma()
'''
随机hsv变换
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''
def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    # hue_delta = np.random.randint(-hue_vari, hue_vari)
    hue_delta = 0
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    sat_mult = 1
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    print(val_mult)
    # val_mult = 1
    return hsv_transform(img, hue_delta, sat_mult, val_mult)

# 虚线
def dotted_line(img):
    line_color = np.random.randint(0,40,3)
    line_color = (int(line_color[0]),int(line_color[1]),int(line_color[2]))
    h,w = img.shape[:2]
    if np.random.random() > 0.3:
        random_num = random.uniform(0.7,1)
    else:
        random_num = random.uniform(0,0.3)
    new_h = int(h*random_num)
    # dot number
    mm_unit = random.randint(4,8)
    # num = random.randint(170,200)
    # mm_unit = int(w/num)
    # print(w,num,mm_unit)
    temp_begin = 0
    temp_end = mm_unit
    while True:
        cv2.line(img, (temp_begin, new_h), (temp_end, new_h), line_color, 2)
        temp_begin+=2*mm_unit
        temp_end+=2*mm_unit
        if temp_end>w:
            break
    return img

def speckle(src, noise_type=1):
    img = src.copy()
    img = img.astype(np.float) / 255
    severity = np.random.uniform(0, 0.2)
    # severity = 0.5
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    blur_avg = np.average(blur[:, :, :3], axis=2)
    blur[:, :, 0] = blur_avg
    blur[:, :, 1] = blur_avg
    blur[:, :, 2] = blur_avg
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    img_speck = img_speck * 255
    img_speck = img_speck.astype(np.uint8)
    return img_speck


def gaussian_noise(src, mean=0, sigma=20, noise_type=1):
    img = src.copy()
    img = img.astype(np.float)
    sigma = np.random.randint(sigma)
    if noise_type == 1:
        noise_mat = np.random.normal(loc=mean, scale=sigma, size=img.shape)
    else:
        noise_mat = np.random.normal(loc=mean, scale=sigma, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

def pepperand_salt_noise(src, percetage=0.05, noise_type=1):
    img = src.copy()
    percetage = np.random.random() * percetage
    # 1:盐噪声, 2:椒噪声
    if noise_type == 1:
        noise_mask = np.random.choice([0, 1, 2], size=img.shape, p=[1 - percetage, percetage / 2, percetage / 2])
    else:
        noise_mask = np.random.choice([0, 1, 2], size=img.shape[:2], p=[1 - percetage, percetage / 2, percetage / 2])
        if len(img.shape) == 3:
            noise_mask = np.expand_dims(noise_mask, axis=-1)

    img = np.where(noise_mask == 1, 255, img)
    img = np.where(noise_mask == 2, 0, img)
    return img

def poisson_noise(src, lam=250, noise_type=1):
    img = src.copy()
    img = img.astype(np.float)
    lam = np.random.randint(lam)
    if noise_type == 1:
        noise_mat = np.random.poisson(lam=lam, size=img.shape)
    else:
        noise_mat = np.random.poisson(lam=lam, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat - np.average(noise_mat)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

def gamma_noise(src, shape=200, scale=1.0, noise_type=1):
    img = src.copy()
    img = img.astype(np.float)
    shape = np.random.randint(shape)
    if noise_type == 1:
        noise_mat = np.random.gamma(shape=shape, scale=scale, size=img.shape)
    else:
        noise_mat = np.random.gamma(shape=shape, scale=scale, size=img.shape[:2])
        if len(img.shape) == 3:
            noise_mat = np.expand_dims(noise_mat, axis=-1)
    img += noise_mat - np.average(noise_mat)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

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

def smooth(image, char_size):
    # print(char_size, image.shape)
    if np.random.choice([True, False]):
        size = int(char_size / 24)
        image = cv2.blur(image, (size * 2 + 1, size * 2 + 1))
    h, w = image.shape[:2]
    scale = 1 - np.random.random() * 0.2
    image = cv2.resize(image, (int(w * scale), int(h * scale)))
    image = cv2.resize(image, (w, h))
    return image

def crop_image(image, point_list=None):
    alpha_img = image[:, :, 3].copy()

    pos = np.where(alpha_img[:, :] > 100)
    x1, x2, y1, y2 = min(pos[1]), max(pos[1]), min(pos[0]), max(pos[0])
    # print('坐标',x1,x2,y1,y2)
    image = image[y1:y2 + 1, x1:x2 + 1, :]
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    h, w = image.shape[:2]
    # print(h,w,'123123')
    w_offset_range = int(h)
    h_offset_range = int(h * 0.3)
    # h_offset_range = int(w)
    # w_offset_range = int(w * 0.3)
    if w_offset_range > 0:
        left, right = np.random.randint(0, w_offset_range, 2)
    else:
        left, right = 0, 0
    if h_offset_range > 0:
        top, bottom = np.random.randint(0, h_offset_range, 2)
    else:
        top, bottom = 0, 0
    new_image = np.zeros((top + bottom + h, left + right + w, 4), dtype=np.uint8)
    new_image[top:top + h, left:left + w, :] = image
    if point_list is not None:
        point_list = [[x - x1 + left, y - y1 + top] for x, y in point_list]
    # print(new_image.shape)
    # print('point_list',point_list)


    # w_new_image = new_image.shape[1]
    # if point_list is not None:
    #     point_list = [[y,w_new_image-x]for x,y in point_list]

    # image_tt = new_image_90[:,:,:].copy()
    # pts = np.array(point_list,np.int32)
    # cv2.polylines(image_tt,[pts],True,(0,0,255))
    # cv2.imshow('image', image_tt)
    # cv2.waitKey(0)
    return new_image, point_list


def add_background(image, bg_image):
    fg_height, fg_width = image.shape[:2]
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    bg_image = Image.fromarray(bg_image)
    bg_image = bg_image.resize((fg_width, fg_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(image)
    r, g, b, a = pil_img.split()
    bg_image.paste(pil_img, (0, 0, fg_width, fg_height), mask=a)
    image = np.array(bg_image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def add_background_with_path(image, bg_path):
    fg_height, fg_width = image.shape[:2]
    bg_image = Image.open(bg_path)
    bg_image = bg_image.resize((fg_width, fg_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(image)
    r, g, b, a = pil_img.split()
    bg_image.paste(pil_img, (0, 0, fg_width, fg_height), mask=a)
    image = np.array(bg_image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image