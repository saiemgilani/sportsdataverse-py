from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from sportsdataverse.cfb import *
from sportsdataverse.mbb import *
from sportsdataverse.nba import *
from sportsdataverse.nfl import *
from sportsdataverse.nhl import *
from sportsdataverse.wbb import *
from sportsdataverse.wnba import *
