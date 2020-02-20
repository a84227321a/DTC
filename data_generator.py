import os

OUTPUT_DIR =''
class TextImageGenerator():
    def __init__(self, char_idx_dict,batch_size,img_h,max_string_len):
        super().__init__()
        self.batch_size = batch_size
        self.img_h = img_h
        self.char_idx_dict = char_idx_dict
        self.chars_list = list(self.char_idx_dict.keys())
        self.blank_label = self.output_size() - 1
        self.max_string_len = max_string_len
    def output_size(self):
        return len(self.char_idx_dict) + 1

    def get_batch(self, size):
        pass
class ValGenerator():
    def __init__(self,base_dir,batch_size,prediction_model,img_h,label_len,characters):
        self.output_dir = os.path.join(
            OUTPUT_DIR, base_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.base_model = prediction_model
        self.img_h = img_h
        self.class_str = characters
        self.batch_size = batch_size
        self.label_len = label_len
    def acc(self):
        pass

