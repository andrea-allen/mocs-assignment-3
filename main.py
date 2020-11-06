import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib import rc
import problem2

if __name__ == '__main__':
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    print('Mocs Assignment 3\n By Mariah Boudreau, Amanda Bertschinger, Andrea Allen')
    problem2.run_model()
