import argparse
import string
import multiprocessing

parser = argparse.ArgumentParser()

parser.add_argument('--height', type=int, default=32)
parser.add_argument('--bg_image_path', type=str, default='bg_images')
parser.add_argument('--nb_epochs', type=int, default=100)
parser.add_argument('--val_iter_period', type=int, default=10000)
parser.add_argument('--gpus', type=int, nargs='*', default=[0])
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--absolute_max_string_len', type=int, default=50)
parser.add_argument('--nb_channels', type=int, default=3)
parser.add_argument('--load_model_path', type=str, default='')

parser.add_argument('--label_len', type=int, default=62)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--absolute_max_string_len', type=int, default=50)
parser.add_argument('--stn', type=str, default= True)
parser.add_argument('--dropout_rate', type=float, default=0.25)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_reduction_factor', type=float, default=0.1)
parser.add_argument('--nb_workers', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--resume_training', type=bool, default=False)

parser.add_argument('--characters', type=str, default='0123456789'+string.ascii_lowercase+'-')
parser.add_argument('--base_dir', type=str, default='data')
parser.add_argument('--output_dir', type=str, default='result')
# parser.add_argument('--save_best_only', type=bool, default=False)
#parser.add_argument('--training_model_cp_filename', type=str, default='model.{epoch:03d}-{loss:.2f}.hdf5')
#parser.add_argument('--prediction_model_cp_filename', type=str, default='prediction_model.{epoch:03d}.hdf5')
# parser.add_argument('--tb_log', type=str, default='tb_log')

cfg = parser.parse_args()
