import logging

logging.basicConfig(format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)

from .merf import MERF
from .utils import MERFDataGenerator
