__author__ = 'po'
from collections import deque
import numpy as np
import math

class CustomEnv():
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # initial balance
        self.balance = 1000
        self.actionSpace = 4  # buy, hold, sell, close
        self.observationSpace = len(dataframe.columns)+1  # ohlcv + net position aka longshortflag
        self.indexPointer = 0
        self.positions = []
        self.fxRate = 1.36
        self.lotSize = 0.1
        self.stopLoss = 50
        self.takeProfit = 100
        self.longShortFlag = 0  # neutral 0 long 1 short -1
        self.logging=False


    def step(self, action):

        self._take_action(action)

        # reward engineering is reward balance?
        reward = self._get_reward()
        ob = self._getState()

        # done when account blowup or reached eod
        episode_over = self.balance <= 0 or self.indexPointer == len(self.dataframe.index)-1

        return ob, reward, episode_over, {}


    def reset(self):
        self.indexPointer = 0
        self.balance=1000
        return self._getState()

    def render(self, mode='human', close=False):
        pass

    def _getState(self):
        return np.append(self.dataframe.iloc[self.indexPointer, :].values, self.longShortFlag)

    def _checkInitialPositions(self, currentState):
        numPositionToClear = 0
        for position in self.positions[:]:

            # cut loss
            # long position and position - currentLow >= stoploss
            if self.longShortFlag == 1 and (position - currentState[2] >= self.stopLoss):
                if(self.logging):print("Long position, Entry: {}, Stoploss: {}, CurrentLow: {}".format(position, self.stopLoss, currentState[2]))
                self.balance -= self.stopLoss * self.fxRate * self.lotSize
                self.positions.remove(position)
                #numPositionToClear +=1
                continue
            # short position and position + highesthigh >= stoploss
            elif self.longShortFlag == -1 and (currentState[1] - position >=  self.stopLoss):
                if(self.logging):print("Short position, Entry: {}, Stoploss: {}, CurrentHigh: {}".format(position, self.stopLoss, currentState[1]))
                self.balance -= self.stopLoss * self.fxRate * self.lotSize
                self.positions.remove(position)
                #numPositionToClear +=1
                continue


            # take profit
            # long position and highesthigh - position >= takeprofit
            if self.longShortFlag == 1 and (currentState[1] - position >= self.takeProfit):
                if(self.logging):print("Long position, Entry: {}, TakeProfit: {}, CurrentHigh: {}".format(position, self.takeProfit, currentState[1]))
                self.balance += self.takeProfit * self.fxRate * self.lotSize
                self.positions.remove(position)
                #numPositionToClear +=1
                continue
            # short position and position - lowestlow >= takeprofit
            elif self.longShortFlag == -1 and (position - currentState[2] >= self.takeProfit):
                if(self.logging):print("Short position, Entry: {}, TakeProfit: {}, CurrentLow: {}".format(position, self.takeProfit, currentState[2]))
                self.balance += self.takeProfit * self.fxRate * self.lotSize
                self.positions.remove(position)
                #numPositionToClear +=1
                continue

        #for _ in range(numPositionToClear):
         #   self.positions.popleft()


    def _take_action(self, action):

        # close old position that hits limit
        currentState = self._getState()
        self._checkInitialPositions(currentState)

        self.indexPointer += 1
        # get new state
        newState = self._getState()
        # buy
        if action == 0:
            # double down into current position provided its in same or neutral direction
            if self.longShortFlag==1 or self.longShortFlag==0:
                # add long position for new state open
                self.positions.append(newState[0])
                self.longShortFlag = 1

        # hold
        elif (action == 1):
            # nothing
            pass

        # sell
        elif (action == 2):
            # double down into current position provided its in same or neutral direction
            if self.longShortFlag==-1 or self.longShortFlag==0:
                self.positions.append(newState[0])
                self.longShortFlag = -1

        elif (action == 3):
            for position in self.positions:
                # close position
                if self.longShortFlag==1:
                    self.balance += (newState[0] - position) * self.fxRate * self.lotSize # 11 - 12
                elif self.longShortFlag ==-1:
                    self.balance += (position - newState[0]) * self.fxRate * self.lotSize # 12 - 11

            # clear all position leaving length to 0
            self.positions.clear()
            # reset flag
            self.longShortFlag = 0






    def _get_reward(self):
        # abs(position - averageClose)
        # just the realised balance we should not consider fluctuations
        return self.balance-1000