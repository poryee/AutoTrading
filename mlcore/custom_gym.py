__author__ = 'po'
from collections import deque
import numpy as np


class CustomEnv():
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # initial balance
        self.balance = 1000
        self.actionSpace = 4  # buy, hold, sell, close
        self.observationSpace = len(dataframe.columns)  # ohlcv 5
        self.indexPointer = 0
        self.positions = deque()
        self.fxRate = 1.36
        self.lotSize = 0.1
        self.stopLoss = 25
        self.takeProfit = 50
        self.longShortFlag = 0  # neutral 0 long 1 short -1


    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = self._get_reward()  # reward engineering is reward balance?
        ob = self._getState()
        # done when account blowup, reached eod
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
        for position in self.positions:

            # cut loss
            # long position and position - currentLow >= stoploss
            if self.longShortFlag == 1 and (position - currentState[2] >= self.stopLoss):
                self.balance -= self.stopLoss * self.fxRate * self.lotSize
                numPositionToClear +=1
                continue
            # short position and position + highesthigh >= stoploss
            elif self.longShortFlag == -1 and (currentState[1] - position >=  self.stopLoss):
                self.balance -= self.stopLoss * self.fxRate * self.lotSize
                numPositionToClear +=1
                continue


            # take profit
            # long position and highesthigh - position >= takeprofit
            if self.longShortFlag == 1 and (currentState[1] - position >= self.takeProfit):
                self.balance += self.takeProfit * self.fxRate * self.lotSize
                numPositionToClear +=1
                continue
            # short position and position - lowestlow >= takeprofit
            elif self.longShortFlag == -1 and (position - currentState[2] >= self.takeProfit):
                self.balance += self.takeProfit * self.fxRate * self.lotSize
                numPositionToClear +=1
                continue

        for _ in range(numPositionToClear):
            self.positions.popleft()


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
                    self.balance += (newState[3] - position) # 11 - 12
                elif self.longShortFlag ==-1:
                    self.balance += (position - newState[3]) # 12 - 11

            # clear all position leaving length to 0
            self.positions.clear()
            # reset flag
            self.longShortFlag = 0






    def _get_reward(self):
        # abs(position - averageClose)
        # just the realised balance we should not consider fluctuations
        return self.balance-1000