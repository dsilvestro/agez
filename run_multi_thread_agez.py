import csv, sys
import os,glob
import time
import argparse
import numpy as np
from numpy import *
import multiprocessing


run = 1
list_args = [1234,
             4321,
             1342,
             1423,
             4231]


def my_job(arg):
    cmd = "python3 agez_run_cmd.py -r %s" % (arg)
    if run:
        os.system(cmd)
    else:
        print(cmd)
        

if __name__ == '__main__': 
    pool = multiprocessing.Pool(len(list_args))
    pool.map(my_job, list_args)
    pool.close()
