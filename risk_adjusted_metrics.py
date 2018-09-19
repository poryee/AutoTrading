__author__ = 'po'
'''
Performance metrics to measure performance of strategy with respect to the risk adjusted returns
some strategy might give higher returns but if the draw downs/volatility is a lot higher for that 1 or 2 percent
it might not be worth
Courtesy of TuningFinance: http://www.turingfinance.com/computational-investing-with-python-week-one/
'''

import math
import numpy as np
import numpy.random as nrand

def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)


def dd(returns, tau):
    # returns the draw-down fiven time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # find max drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i

        pos, pre = pos - 1, pre - 1
    # drawdown should be pos
    return abs(drawdown)


def max_dd(returns):
    # returns the max draw-down for any tau in (0, T), where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # max draw-down should be positive
    return abs(max_drawdown)


#  A higher Calmar Ratio suggests more returns at lower risk.
def calmar_ratio(er, returns, rf):
    return (er-rf) / max_dd(returns)
