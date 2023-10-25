import torch
# from pytorch_transformers import *
import sys,logging
logging.root.handlers = []
logging.basicConfig(level="INFO", format = '%(asctime)s:%(levelname)s: %(message)s' ,stream = sys.stdout)
logger = logging.getLogger(__name__)
logger.info('hello')
print(torch.cuda.is_available(), torch.cuda.device_count())


def check_memory():
    logger.info('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))