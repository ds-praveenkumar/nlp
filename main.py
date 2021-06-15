# main.py
import torch 

from logger import get_basic_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = get_basic_logger()

log.info( f'torch version: {torch.__version__ }' )
log.info( f'device: {device}')

