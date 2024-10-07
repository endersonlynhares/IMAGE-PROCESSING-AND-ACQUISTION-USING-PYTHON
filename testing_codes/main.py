import time
from numpy import *

def main():
    noofterms = 10000000
    den = linspace(1, noofterms*2, noofterms)
    num = ones(noofterms)
    for i in range(1, noofterms):
        num[i] = pow(-1, i)

    counter = 0
    sum_value = 0

    t1 = time.perf_counter()
    while counter < noofterms:
        sum_value = sum_value + (num[counter] / den[counter])
        counter += 1
    pi_value = sum_value * 4
    print(f'pi_value = {pi_value:2f}')
    t2 = time.perf_counter()
    timetaken = t2 - t1
    print(f'timetaken = {timetaken}')

if __name__ == '__main__':
    main()