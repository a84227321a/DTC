import os
import numpy as np
import shutil
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Dense, LSTM, add, concatenate, \
    Dropout, Lambda, Flatten
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import resnet
from utils import  MultiGPUModelCheckpoint, PredictionModelCheckpoint, Evaluator, create_result_subdir
from stn_utils import STN
from data_generator import TextImageGenerator,ValGenerator
from config import cfg
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    return loss

def model_STN(cfg):
    if K.image_data_format() == 'channels_first':
        # input_shape = (cfg.nb_channels, cfg.height, cfg.width)
        input_shape = (cfg.nb_channels, cfg.height, None)
    else:
        # input_shape = (cfg.height, cfg.width, cfg.nb_channels)
        input_shape = (cfg.height, None, cfg.nb_channels)
    inputs_data = Input(name='the_input', shape=input_shape, dtype='float32')
    if cfg.stn:
        if K.image_data_format() == 'channels_first':
            x = STN(inputs_data, sampling_size=input_shape[1:])
        else:
            x = STN(inputs_data, sampling_size=input_shape[:2])
    y_pred = resnet.ResNet50(x, len(cfg.characters))
    prediction_model = Model(inputs=inputs_data, outputs=y_pred)
    prediction_model.summary()
    labels = Input(name='the_labels', shape=[cfg.label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    ctc_loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])
    training_model = Model(inputs=[inputs_data, labels, input_length, label_length],
                  outputs=[ctc_loss_out])
    return training_model, prediction_model

def get_generators():
    # train_generator = TrainGenerator(base_dir=cfg.base_dir,
    #                                  annotation_file=os.path.join(cfg.base_dir, 'annotation_train.txt'),
    #                                  batch_size=cfg.batch_size,
    #                                  img_size=(cfg.width, cfg.height),
    #                                  nb_channels=cfg.nb_channels,
    #                                  timesteps=cfg.timesteps,
    #                                  label_len=cfg.label_len,
    #                                  characters=cfg.characters)
    train_generator = TextImageGenerator(char_idx_dict=cfg.batch_size,
                                 batch_size=cfg.height,
                                 prediction_model= prediction_model,
                                 img_h=cfg.height,
                                 absolute_max_string_len=cfg.absolute_max_string_len
                                 )
    val_generator = ValGenerator(base_dir=cfg.base_dir,
                                 batch_size=32,
                                 img_h=cfg.height,
                                 label_len=cfg.label_len,
                                 characters=cfg.characters)
    return train_generator, val_generator

def get_optimizer():
    if cfg.optimizer == 'sgd':
        opt = SGD(lr=cfg.lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    elif cfg.optimizer == 'adam':
        opt = Adam(lr=cfg.lr)
    else:
        raise ValueError('Wrong optimizer name')
    return opt

def create_output_directory():
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_subdir = create_result_subdir(cfg.output_dir)
    print('Output directory: ' + output_subdir)
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        shutil.copy(f, output_subdir)
    return output_subdir

def get_callbacks(output_subdir, training_model, prediction_model, val_generator):
    training_model_checkpoint = MultiGPUModelCheckpoint(os.path.join(output_subdir, cfg.training_model_cp_filename), training_model, save_best_only=cfg.save_best_only, monitor='loss', mode='min')
    prediction_model_checkpoint = PredictionModelCheckpoint(os.path.join(output_subdir, cfg.prediction_model_cp_filename), prediction_model, save_best_only=cfg.save_best_only, monitor='loss', mode='min')
    evaluator = Evaluator(prediction_model, val_generator, cfg.label_len, cfg.characters, cfg.optimizer, period=cfg.val_iter_period)
    lr_reducer = ReduceLROnPlateau(factor=cfg.lr_reduction_factor, patience=3, verbose=1, min_lr=0.00001)
    os.makedirs(cfg.tb_log, exist_ok=True)
    tensorboard = TensorBoard(log_dir=cfg.tb_log)
    return [training_model_checkpoint, prediction_model_checkpoint, evaluator, lr_reducer, tensorboard]

def load_weights_if_resume_training(training_model):
    if cfg.resume_training:
        training_model.load_weights(cfg.load_model_path)
    return training_model

if __name__ == '__main__':
    training_model, prediction_model = model_STN(cfg)
    output_subdir = create_output_directory()
    opt = get_optimizer()
    train_generator, val_generator = get_generators()
    training_model.compile(loss={'ctc': lambda y_true, ctc_pred: ctc_pred}, optimizer=opt, metrics=['accuracy'])
    callbacks = get_callbacks(output_subdir, training_model, prediction_model, val_generator)
    # use the model
    training_model = load_weights_if_resume_training(training_model)
    training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    print(training_model.optimizer)
    print("Learning rate: " + str(K.eval(training_model.optimizer.lr)))
    training_model.fit_generator(train_generator,
                                 steps_per_epoch=10000,
                                 epochs=cfg.nb_epochs,
                                 verbose=1,
                                 workers=cfg.nb_workers,
                                 use_multiprocessing=True,
                                 callbacks=callbacks)


