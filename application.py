from mlcore.custom_gym import CustomEnv

__author__ = 'po'

# scheduler
from datetime import datetime, timedelta

# plot indicator charts
import matplotlib.pyplot as plt

# ig services
from dataprovider.ig_service import IGService
# defines username, password, api_key, acc_type
from dataprovider.ig_service_config import *
# libs
from risk_adjusted_metrics import *
from mlcore.rl_agent import torchDQN
import glob
import pandas as pd
import collections

# fetch the open high low and close at daily time resolution for the past 20 days for analysis
def getHistoricalData(specificDate):
    ig_service = IGService(username, password, api_key, acc_type)
    ig_service.create_session()

    '''
    # dynamically retrieve the epic/product code
    print("fetch_all_watchlists")
    response = ig_service.fetch_all_watchlists()
    # get "MyWatchlist"
    watchlist_id = response['id'].iloc[2]

    print("fetch_watchlist_markets")
    response = ig_service.fetch_watchlist_markets(watchlist_id)
    print(response)
    epic = response['epic'].iloc[0]

    print("fetch_market_by_epic")
    response = ig_service.fetch_market_by_epic(epic)
    print(response)
    '''

    print("search_pricing")
    # Instrument tag # US500 DFB
    epic = 'IX.D.DOW.IGD.IP'

    # Price resolution (SECOND, MINUTE, MINUTE_2, MINUTE_3, MINUTE_5, MINUTE_10, MINUTE_15, MINUTE_30, HOUR, HOUR_2, HOUR_3, HOUR_4, DAY, WEEK, MONTH)
    resolution = 'MINUTE_5'  # resolution = 'H', '1Min'



    # (yyyy:MM:dd-HH:mm:ss)
    # today = datetime.today()
    # startDate = str(today.date() - timedelta(days=2)).replace("-", ":") + "-00:00:00"
    #endDate = str(today.date() - timedelta(days=0)).replace("-", ":") + "-00:00:00"

    startDate = specificDate.date().strftime('%Y:%m:%d') + "-00:00:00"
    endDate = (specificDate + timedelta(days=1)).date().strftime('%Y:%m:%d') + "-23:55:00"

    response = ig_service.fetch_historical_prices_by_epic_and_date_range(epic, resolution, startDate, endDate)
    return response['prices']


def bulkDownload(date, numberOfDays):
    specificDate = datetime.strptime(date, '%Y-%m-%d')

    # 10 days
    for i in range(numberOfDays):
        if (specificDate.weekday() != 6):
            pastData = getHistoricalData(specificDate)
            saveSpecificDate(pastData, specificDate)
        specificDate -= timedelta(days=1)


def getAverage(dataArray):
    tempList = []
    for priceObject in dataArray:
        if (priceObject['bid'] != None and priceObject['ask'] != None):
            tempList.append(round((priceObject['ask'] + priceObject['bid']) / 2, 2))
        else:
            tempList.append(0)
    return tempList


def saveSpecificDate(pastData, date):
    # iterate list of json and average up the results
    pastData['averageOpen'] = getAverage(pastData['openPrice'])
    pastData['averageLow'] = getAverage(pastData['lowPrice'])
    pastData['averageHigh'] = getAverage(pastData['highPrice'])
    pastData['averageClose'] = getAverage(pastData['closePrice'])

    pastData.to_csv("data/" + str(date.date()) + ".csv")


# construct stochastic indicator to determine momentum in direction using (formula is just highest high - close/ highest high - lowest low)* 100 to get percentage
def constructIndicator(pastData):
    # http://www.andrewshamlet.net/2017/07/13/python-tutorial-stochastic-oscillator/
    # http://www.pythonforfinance.net/2017/10/10/stochastic-oscillator-trading-strategy-backtest-in-python/

    # Create the "lowestLow" column in the DataFrame
    pastData['lowestLow'] = pastData['averageLow'].rolling(window=14).min()

    # Create the "highestHigh" column in the DataFrame
    pastData['highestHigh'] = pastData['averageHigh'].rolling(window=14).max()

    # Create the "%K" column in the DataFrame refer to the function comment for formula of stochastic ociliator
    pastData['%K'] = ((pastData['averageClose'] - pastData['lowestLow']) / (
        pastData['highestHigh'] - pastData['lowestLow'])) * 100


    # Create the "%D" column in the DataFrame moving average of calculated K
    pastData['%D'] = pastData['%K'].rolling(window=3).mean()

    # drop 14 bar ago cut away parts of chart without indicator
    # pastData.drop(pastData.index[:15], inplace=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    pastData['averageClose'].plot(ax=axes[0]);
    axes[0].set_title('Close')
    pastData[['%K', '%D']].plot(ax=axes[1]);
    axes[1].set_title('Oscillator')
    # plt.show()

    return pastData
    # consider building other indicator


# measure and evaluate system developed
def performanceTest(returns):
    # Calmar
    # The Calmar ratio discounts the expected excess return of a portfolio by the worst expected maximum draw down for that portfolio,

    # simulation
    # Returns from the portfolio (r) and market (m)
    # returns = nrand.uniform(-1, 1, 50)

    # Expected return
    averageExpectedReturn = np.mean(returns)

    # Risk Free Rate assumption that 6% from other investment as benchmark (Opportunity cost)
    riskFreeRate = 0.06

    print("Calmar Ratio =", calmar_ratio(averageExpectedReturn, returns, riskFreeRate))


# retrieve all downloaded data concatenated in dataframe
def retrievePastDataDataframe():
    # retrieve all csv in data folder
    allFiles = glob.glob("data/*.csv")

    # concat csv
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, sep=',', index_col=0, header=0)
        list_.append(df)
    # concatenate every row in the list and reset index to sequential
    pastData = pd.concat(list_, axis=0, ignore_index=True)

    # setsnapshotTime as index
    pastDataAsState = pastData[
        ['snapshotTime', 'averageOpen', 'averageHigh', 'averageLow', 'averageClose', 'lastTradedVolume']]
    pastDataAsState.snapshotTime = pd.to_datetime(pastData['snapshotTime'], format='%Y:%m:%d-%H:%M:%S')
    pastDataAsState.set_index('snapshotTime', inplace=True)

    # print(pastDataAsState.info())
    # asd=pastDataAsState.iloc[0,:]
    # print(pastDataAsState.loc['2019-02-01'])
    # print(type(pastDataAsState.loc['2019-02-01']))

    return pastDataAsState


def resampleDataframe(dataframe, timeframe):
    pd.set_option('display.max_columns', None)

    data = dataframe.resample(timeframe).agg({'averageOpen': 'first',
                                              'averageHigh': 'max',
                                              'averageLow': 'min',
                                              'averageClose': 'last',
                                              'lastTradedVolume': 'sum'
    })
    data.dropna(inplace=True)
    return data


def visualise(dataframe, episodeAction, episodeReward):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

    dataframe.plot(y="averageClose", ax=ax1)
    buyTime, buyPrice, sellTime, sellPrice = [], [], [], []
    for index, action in enumerate(episodeAction):
        if (action == 0):
            buyTime.append(dataframe.index[index])
            buyPrice.append(dataframe.averageClose[index])
        elif action == 2:
            sellTime.append(dataframe.index[index])
            sellPrice.append(dataframe.averageClose[index])

    ax1.scatter(buyTime, buyPrice, c='g', marker="^", s=25)
    ax1.scatter(sellTime, sellPrice, c='r', marker="v", s=25)

    ax2.plot(episodeReward)
    fig.subplots_adjust(hspace=1)

    plt.show()


def trainMLModel(endDate, timeResolution, trainingDays, totalEpisodes):
    # retrieve pastdata
    pastDataAsState = retrievePastDataDataframe()
    pastDataAsState = resampleDataframe(pastDataAsState, timeResolution)

    dqn = torchDQN(tensorboard=True)
    total_reward = []
    total_action = []
    total_steps = 0

    endDate = datetime.strptime(endDate, '%Y-%m-%d')

    # because we want to include endDate in the training hence -1
    traingDate = endDate - timedelta(days=trainingDays - 1)

    # so we end up with largest quotient when dividing totalEpisodes with trainingDays so we end training on endDate
    trainingEpisode = (totalEpisodes // trainingDays) * trainingDays

    print('\nCollecting experience...')
    for i_episode in range(trainingEpisode):
        if (i_episode % (trainingDays) == 0):
            # because we want to include endDate in the training hence -1
            traingDate = endDate - timedelta(days=trainingDays - 1)

        if (pastDataAsState[traingDate.date().strftime('%Y-%m-%d')].empty):
            traingDate += timedelta(days=1)
            continue

        # initialise gym environment with single day slice of past data
        env = CustomEnv(pastDataAsState.loc[traingDate.date().strftime('%Y-%m-%d')])

        # multiday training do we reset the balance?
        s = env.reset()
        ep_r = 0
        while True:
            # env.render()
            # see how random trading with 2:1 RRR will perform
            # a = np.random.randint(0, 4)

            a = dqn.choose_action(s)
            total_action.append(a)
            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2

            dqn.store_transition(s, a, r, s_)
            ep_r += r

            # every 10k steps we train our model both eval and target
            if dqn.memory_counter > 10000:
                # but target updates at a slower rate so learning is more stable
                # think of eval as the hyper active child and target as the parent that critics the child exploration
                dqn.learn(total_steps)

            if done:
                print('Ep: ', i_episode, '| Training Date: ', traingDate.date().strftime('%Y-%m-%d'), '| Ep_r: ',
                      round(r, 2))
                ep_r = r
                break
            s = s_
            total_steps+=1

        total_reward.append(ep_r)
        traingDate += timedelta(days=1)

    counter = collections.Counter(total_action)
    print("total unique action ", print(counter))

    # display visualisation of training result
    plt.title('Reward')
    plt.xlabel('No of Episodes')
    plt.ylabel('Total reward')
    plt.plot(np.arange(len(total_reward)), total_reward, 'r-', lw=5)
    plt.show()

    # save trained model
    dqn.save()


def evaluateMLModel(evalutionDate, timeResolution="5min", showChart=False):
    # retrieve past data
    pastDataAsState = retrievePastDataDataframe()
    pastDataAsState = resampleDataframe(pastDataAsState, timeResolution)

    # initialise gym environment with single day slice of past data
    env = CustomEnv(pastDataAsState.loc[evalutionDate])

    dqn = torchDQN()
    totalFinalReward = []
    totalAction = []
    print('\nCollecting experience...')

    # trade the same day 10 times
    for i_episode in range(100):
        s = env.reset()
        episodeAction = []
        episodeBalance = [env.balance]
        while True:
            # env.render()
            # see how random trading with 2:1 RRR will perform
            # a = np.random.randint(0, 4)

            a = dqn.choose_action(s)
            episodeAction.append(a)
            # take action
            s_, r, done, info = env.step(a)

            # store balance note that reward is balance - initial capital hence 0
            episodeBalance.append(env.balance)

            # no training for evaluation

            if done:
                print('Ep: ', i_episode, '| Ep_r: ', round(r, 2))
                if (showChart == True):
                    visualise(env.dataframe, episodeAction, episodeBalance)

                break
            s = s_
        # collect stats
        totalFinalReward.append(r)
        totalAction.extend(episodeAction)

    counter = collections.Counter(totalAction)
    print("total unique action ", print(counter))

    plt.title('Reward')
    plt.xlabel('No of Episodes')
    plt.ylabel('Total Final Reward')
    plt.plot(np.arange(len(totalFinalReward)), totalFinalReward, 'r-', lw=5)

    # Calculate the simple average of the rewards
    yMean = [np.mean(totalFinalReward)] * len(totalFinalReward)
    plt.plot(yMean, label='Mean', linestyle='--')
    plt.text(1, yMean[0], "Average Returns: {:.3f}".format(yMean[0]))
    plt.show()

    winloseRatio = len(list(filter(lambda x: x > 0, totalFinalReward))) / len(totalFinalReward)
    print(winloseRatio)
    # temporary placeholder for balance
    return True


def automateTrading():
    pass
    # pastDataWithIndicator = constructIndicator(pastData)

    # using past 20 day data
    # machineLearning(pastData)


'''
what is the key outcome?
1) reinforcement to trade profitably daily basis <-- Done DDQN will trt rdn when got time
2) robust when back tested against historical data 2 month <-- Done
3) automate trade demo using model and algorithm (completed custom gym environment for agent to interact with based on ig dow jones data in 5min resolution)
4) unrealised profit or loss into state & new action close position <-- Done
5) Multiday training and validation <-- Done only slight gains from 25points sl 50tp 2:1 RRR
6) resample to high time frame for evaluation (Check if need to do that for training as well) <-- Done (so that we don't have to trade that late at night)
7) custom env to provide returns array via info for performanceTest (Optimise hyper param) 1 month training plus 1 week walk forward
    (have yet to test other point base sl:tp, e.g 40:80 okok but must make sure loss is low, 50:100 seems lit, 25:75 cmi, 50:25 not worth the drawdowns)
    (100:50 crazy drawdown on certain days, 150:50 high winrate but not much profit a few losses would wipe the floor) <-- seems like scalping isnt the way to go here
8) check if underfit or overfit model <-- Done added tensorboard to monitor loss overtime noticed that beyond 20k steps aka 600ish episode the model starts to diverge
9) Trade profiler for evaluation Average High/Low + Max High/Low, Win/Lose Rate, RRR
10) https://www.kaggle.com/itoeiji/deep-reinforcement-learning-on-stock-data
11) check for shitty data 0.0,0.0,0.0,0.0 ctrl+shift+f
'''
if __name__ == "__main__":
    #bulkDownload('2019-05-08', 4)
    #trainMLModel(endDate="2019-04-13", timeResolution="30min", trainingDays=30, totalEpisodes=600)

    # exactly the same steps as trainMLModel but without saving while loading trained model
    results = evaluateMLModel(evalutionDate="2019-04-17", timeResolution="30min", showChart=False)
    # performanceTest(results)list(filter(lambda x: x >0, nums))

    # automateTrading()
    # TODO after all is set and done final todo is to use it on CFD account