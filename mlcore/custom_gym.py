__author__ = 'po'
from collections import deque


class CustomEnv():
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # initial balance
        self.balance = 1000
        self.actionSpace = 3  # buy, do nothing, sell (for simplicity we let fixed sl tp dictate exit)
        self.observationSpace = len(dataframe.columns)  # ohlcv 5
        self.indexPointer = 0
        self.positions = deque()
        self.fxRate = 1.36
        self.lotSize = 0.1
        self.stopLoss = 25
        self.takeProfit = 50
        self.longShortFlag = 0  # 0 long 1 short


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
        return self.dataframe.iloc[self.indexPointer, :].values

    def _clearPositions(self, currentState, action):
        positionToClear = 0
        for position in self.positions:

            # cut loss
            # long position - stoploss < lowestlow
            if self.longShortFlag == 0 and (position - self.stopLoss) < currentState[2]:
                self.balance -= self.stopLoss * self.fxRate * self.lotSize
                positionToClear +=1
                continue
            # short position + stoploss > highesthigh
            elif self.longShortFlag == 1 and (position + self.stopLoss) > currentState[1]:
                self.balance -= self.stopLoss * self.fxRate * self.lotSize
                positionToClear +=1
                continue


            # take profit
            # long position + takeprofit > highesthigh
            if self.longShortFlag == 0 and (position - self.stopLoss) < currentState[1]:
                self.balance += self.takeProfit * self.fxRate * self.lotSize
                positionToClear +=1
                continue
            # short position - stoploss < lowestlow
            elif self.longShortFlag == 1 and (position + self.stopLoss) > currentState[2]:
                self.balance += self.takeProfit * self.fxRate * self.lotSize
                positionToClear +=1
                continue




            # close position
            if action == 0 and self.longShortFlag==1:
                self.balance += (currentState[3] - position) # 11 - 12
                positionToClear +=1
                continue
            elif action == 2 and self.longShortFlag ==0:
                self.balance += (position - currentState[3]) # 12 - 11
                positionToClear +=1

        for _ in range(positionToClear):
            self.positions.popleft()

    def _addPositions(self, newState, action):
        if action == 0:
            # add long position for new state open
            self.positions.append(newState[0])
            self.longShortFlag = 0

        elif (action == 1):
            # nothing
            pass
        else:
            self.positions.append(newState[0])
            self.longShortFlag = 0
        pass

    def _take_action(self, action):
        currentState = self._getState()
        self.indexPointer += 1
        self._clearPositions(currentState, action)

        newState = self._getState()
        self._addPositions(newState, action)




    def _get_reward(self):
        # abs(position - averageClose)
        # just the realised balance we should not consider fluctuations
        return self.balance