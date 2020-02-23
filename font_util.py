import curses.ascii
import os
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle


class FontHelper:
    def __init__(self, font_dir):
        self.__font_dir = font_dir
        self.__font_chars = list()
        self.__load_font()

    @staticmethod
    def __get_chars(font_path):
        ttf = TTFont(font_path)
        tables = ttf['cmap'].tables
        all_chars = [char for area in tables for char in area.cmap.keys()]
        chars = list()
        font = ImageFont.truetype(font_path, 26)
        for char_code in all_chars:
            if not curses.ascii.iscntrl(char_code):
                char = chr(char_code)
                ori_text_image = Image.new("RGB", (60, 60), (255, 255, 255))
                draw_handle = ImageDraw.Draw(ori_text_image, "RGB")
                draw_handle.text((3, 3), char, fill=(0, 0, 0), font=font)
                text_image = np.asarray(ori_text_image)
                if np.min(text_image) < 128 or char_code == 32 or char_code == 12288 or char_code == 160:
                    chars.append(char)
        return chars

    def __load_font(self):
        pkl_path = os.path.join(self.__font_dir, 'font_chars.pkl')
        if os.path.exists(pkl_path):
            self.__font_chars = list()
            with open(pkl_path, 'rb') as file:
                font_chars = pickle.load(file)
                for font_path, char_set in font_chars:
                    # font_name = os.path.basename(font_path)
                    font_name = font_path.split('\\')[-1]
                    font_path = os.path.join(self.__font_dir, font_name)
                    self.__font_chars.append([font_path, char_set])
        else:
            self.__font_chars = list()

            font_name_list = os.listdir(self.__font_dir)
            for font_idx, font_name in enumerate(font_name_list):
                if font_name.endswith('.pkl'):
                    continue
                font_path = os.path.join(self.__font_dir, font_name)
                try:
                    chars = self.__get_chars(font_path)
                    char_set = set(chars)
                    self.__font_chars.append([font_path, char_set])
                except Exception as e:
                    pass
                print('font_idx', font_idx)
            with open(pkl_path, 'wb') as file:
                pickle.dump(self.__font_chars, file)
                # print(font_path, char_set)

    @staticmethod
    def __text_in_font(text, char_set):
        for char in text:
            if char not in char_set:
                return False

        return True

    def get_fonts(self, text):
        fonts = list()
        for font_path, char_set in self.__font_chars:
            if self.__text_in_font(text, char_set):
                fonts.append(font_path)
        return fonts
