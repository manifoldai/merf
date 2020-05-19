import logging

from .merf import MERF
from .utils import MERFDataGenerator
from .viz import plot_merf_training_stats

logging.basicConfig(format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)


# Version of the merf package
__version__ = "1.0"
