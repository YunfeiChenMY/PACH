import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

# dataset can be 'WIKI', 'MIRFlickr' or 'NUSWIDE'
DATASET = 'MSCOCO'

if DATASET == 'WIKI':

    DATA_DIR = r'C:\data_diy\dataset\DJSR\wikipedia_dataset/images'
    LABEL_DIR = r'C:\data_diy\dataset\DJSR\wikipedia_dataset/raw_features.mat'
    TRAIN_LABEL = r'C:\data_diy\dataset\DJSR\wikipedia_dataset/trainset_txt_img_cat.list'
    TEST_LABEL = r'C:\data_diy\dataset\DJSR\wikipedia_dataset/testset_txt_img_cat.list'

    BETA = 0.3
    LAMBDA1 = 0.3
    LAMBDA2 = 0.3
    NUM_EPOCH = 201
    LR_IMG = 0.001
    LR_TXT = 0.001
    EVAL_INTERVAL = 50


if DATASET == 'MIRFlickr':

    # LABEL_DIR = r'C:\data_diy\dataset\DJSR\Cleared-Set\LAll/mirflickr25k-lall.mat'
    # TXT_DIR = r'C:\data_diy\dataset\DJSR\Cleared-Set\YAll/mirflickr25k-yall.mat'
    # IMG_DIR = r'C:\data_diy\dataset\DJSR\Cleared-Set\IAll/mirflickr25k-iall.mat'
    
    BETA = 0.9
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 50
    LR_IMG = 0.001
    LR_TXT = 0.001
    EVAL_INTERVAL = 50

if DATASET == 'NUSWIDE':

    # LABEL_DIR = r'C:\data_diy\dataset\DJSR\NUS-WIDE-TC10/nus-wide-tc10-lall.mat'
    # TXT_DIR = r'C:\data_diy\dataset\DJSR\NUS-WIDE-TC10/nus-wide-tc10-yall.mat'
    # IMG_DIR = r'C:\data_diy\dataset\DJSR\NUS-WIDE-TC10\IAll\IAll/nus-wide-tc10-iall.mat'

    BETA = 0.6
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 50
    LR_IMG = 0.001
    LR_TXT = 0.001
    EVAL_INTERVAL = 120

if DATASET == 'MSCOCO':

    # LABEL_DIR = r'C:\data_diy\dataset\DJSR\NUS-WIDE-TC10/nus-wide-tc10-lall.mat'
    # TXT_DIR = r'C:\data_diy\dataset\DJSR\NUS-WIDE-TC10/nus-wide-tc10-yall.mat'
    # IMG_DIR = r'C:\data_diy\dataset\DJSR\NUS-WIDE-TC10\IAll\IAll/nus-wide-tc10-iall.mat'

    BETA = 0.6
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    NUM_EPOCH = 50
    LR_IMG = 0.001
    LR_TXT = 0.001
    EVAL_INTERVAL = 120


BATCH_SIZE = 128
CODE_LEN = 128
MU = 1.5
ETA = 0.4

LR_IMGTXT = 0.001

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

GPU_ID = 0
NUM_WORKERS = 0
EPOCH_INTERVAL = 2

MODEL_DIR = './checkpoint'


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)


logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('BETA = %.4f' % BETA)
logger.info('LAMBDA1 = %.4f' % LAMBDA1)
logger.info('LAMBDA2 = %.4f' % LAMBDA2)
logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)
logger.info('LR_TXT = %.6f' % LR_IMGTXT)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('MU = %.4f' % MU)
logger.info('ETA = %.4f' % ETA)
logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)
logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)
logger.info('--------------------------------------------------------------------')