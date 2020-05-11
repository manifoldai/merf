import logging

from .merf import MERF
from .utils import MERFDataGenerator

logging.basicConfig(format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)


# Version of the merf package
__version__ = "0.3.0"
