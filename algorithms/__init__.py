# path = "/home/cindy/Documents/stage 2020/SIMULATIONS/SelfishUCB_rnd"
# import sys
# if path not in sys.path:
#     sys.path.append(path)
try:
    from .cklucb.cklucb import computeKLUCB
except:
    pass

from .selfishucb import *
from .ecsic import *
from .sicmmab import *
from .sicmmab2 import *
from .lugosi1 import *
from .lugosi2 import *
from .dynmmab import *
from .dynmc import *
from .mctopm import *
from .bubeck import *

# from .dynrnd import *
# from .dynselfishucb import *
# from .musical_chairs import *
#
# from .exp3 import *
# from .egreedy import *
