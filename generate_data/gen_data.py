import cv2
import numpy as np
from PIL import Image, ImageDraw,ImageFont

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
def gen_seal(width,height,font_path,text,theta):
    font_size =20
    font = ImageFont.truetype(font_path, font_size)
    ori_text_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    all_theta =-theta
    for tx in text:
        ori_text_image = ori_text_image.rotate(theta)
        all_theta+=theta
        draw_handle = ImageDraw.Draw(ori_text_image, "RGBA")
        draw_handle.arc([30, 30, 270, 270],  0, 360, fill='red')
        draw_handle.text((width/2, 30), tx, fill=(255,0,0),font=font)
    check_theta = 360-all_theta/2
    ori_text_image = ori_text_image.rotate(check_theta)
    ori_text_image.save("www.png", "png")
    text_image = cv2.cvtColor(np.asarray(ori_text_image), cv2.COLOR_RGBA2BGRA)

    cv2.imshow('a',text_image)
    cv2.waitKey(0)


    # ori_text_image.show()

if __name__ == '__main__':
    width = 300
    height = 300
    theta = 50
    text = '武汉市王麟琦'
    font_path = r'F:\code\GTC\get_data\font\simsun.ttf'
    gen_seal(width,height,font_path,text,theta)