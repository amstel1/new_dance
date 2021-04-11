import logging
import multiprocessing
import numpy as np
from datetime import datetime
N_CPU = 12
RANGE_MIN = 10
RANGE_MAX = int(1e6)


def decompose(inp):
    logger = logging.basicConfig(filename='logfile.log',filemode='a',datefmt='%Y-%m-%d %HH:%MM:%SS',level=logging.INFO,format='%(message)s -- %(asctime)s')
    l = list(str(inp))
    l = [int(x) for x in l]
    s = np.sum(l)
    l.append(s)
    while l[-1]<= inp:
        s = np.sum(l[-3:-1])
        l.append(s + np.sum(l[-1]))
        if l[-1] == inp:
            logging.info('%s success: %s', inp, l)
            break
    #else:
        #pass
        #print('unsucces', inp, l)

def main():
    main_logger = logging.basicConfig(filename='logfile.log',filemode='a',datefmt='%Y-%m-%d %HH:%MM:%SS',level=logging.INFO,format='%(message)s -- %(asctime)s')
    logging.info('started at %s', str(datetime.now()))
    logging.info('number of CPUs at %s', N_CPU)
    logging.info('range from %s to %s', RANGE_MIN, RANGE_MAX)
    with multiprocessing.Pool(N_CPU) as p:
         p.map(decompose, range(RANGE_MIN,RANGE_MAX))
    logging.info('finised at %s', str(datetime.now()))

if __name__ == '__main__':
    main()
    