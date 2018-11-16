__author__ = 'po'

# scheduler
from datetime import datetime, timedelta

# plot indicator charts
import matplotlib.pyplot as plt

# ig services
from ig_service import IGService
# defines username, password, api_key, acc_type
from ig_service_config import *
# libs
from ml_model import AlphaGenerator
from risk_adjusted_metrics import *
from rl_agent import Agent


# fetch the open high low and close at daily time resolution for the past 20 days for analysis
def getHistoricalData():
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
    resolution = 'MINUTE_30'  # resolution = 'H', '1Min'



    # (yyyy:MM:dd-HH:mm:ss)
    today = datetime.today()
    startDate = str(today.date() - timedelta(days=30)).replace("-", ":") + "-00:00:00"
    endDate = str(today.date() - timedelta(days=0)).replace("-", ":") + "-00:00:00"

    response = ig_service.fetch_historical_prices_by_epic_and_date_range(epic, resolution, startDate, endDate)
    return response['prices']


def getAverage(dataArray):
    tempList = []
    for priceObject in dataArray:
        tempList.append((priceObject['ask'] + priceObject['bid']) / 2)
    return tempList


# construct stochastic indicator to determine momentum in direction using (formula is just highest high - close/ highest high - lowest low)* 100 to get percentage
def constructIndicator(pastData):
    # TODO use the market data function to retrieve the OHLC
    # http://www.andrewshamlet.net/2017/07/13/python-tutorial-stochastic-oscillator/
    # http://www.pythonforfinance.net/2017/10/10/stochastic-oscillator-trading-strategy-backtest-in-python/

    # iterate list of json and average up the results
    pastData['averageOpen'] = getAverage(pastData['openPrice'])
    pastData['averageLow'] = getAverage(pastData['lowPrice'])
    pastData['averageHigh'] = getAverage(pastData['highPrice'])
    pastData['averageClose'] = getAverage(pastData['closePrice'])

    # Create the "lowestLow" column in the DataFrame
    pastData['lowestLow'] = pastData['averageLow'].rolling(window=14).min()

    # Create the "highestHigh" column in the DataFrame
    pastData['highestHigh'] = pastData['averageHigh'].rolling(window=14).max()

    # Create the "%K" column in the DataFrame refer to the function comment for formula of stochastic ociliator
    pastData['%K'] = ((pastData['averageClose'] - pastData['lowestLow']) / (
        pastData['highestHigh'] - pastData['lowestLow'])) * 100


    # save so we can retrain model using most recent data
    #pastData.to_csv("yahoo_finance\\30Min.csv", sep=',', encoding='utf-8')

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



# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockClosingDataVec(key):
    vec = []
    lines = open("yahoo_finance/" + key + ".csv", "r").read().splitlines()

    # ignore header
    for line in lines[1:]:
        # append closing price into list
        vec.append(float(line.split(",")[4]))

    return vec

# returns the sigmoid
def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))

    # return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))


    return np.array([res])


# proceed to use machine learning algorithm to predict the weekly price range to provide more confidence
def machineLearning():
    # use RNN to predict the following week
    # https://github.com/LiamConnell/deep-algotrading/blob/master/notebooks/lstm_(7).ipynb
    # https://github.com/Yvictor/TradingGym
    # alphaGenerator = AlphaGenerator()

    # assuming we have better prediction algorithm in the future
    # prediction = alphaGenerator.steadyROI()

    '''
    stock_name, window_size, episode_count = "^GSPC", int(10), int(1000)

    agent = Agent(window_size)
    closingData = getStockClosingDataVec(stock_name)
    l = len(closingData) - 1
    batch_size = 32

    episode_count = 1000
    for e in range(episode_count + 1):
        print ("Episode " + str(e) + "/" + str(episode_count))
        state = getState(closingData, 0,window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state)

            # sit
            next_state = getState(closingData, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buy
                agent.inventory.append(closingData[t])
                print("Buy: " + formatPrice(closingData[t]))

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                reward = max(closingData[t] - bought_price, 0)
                total_profit += closingData[t] - bought_price
                print("Sell: " + formatPrice(closingData[t]) + " | Profit: " + formatPrice(closingData[t] - bought_price))

            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 100 == 0:
            agent.model.save("model_backup/newmodel_ep" + str(e))
    '''

    window_size = 10
    # train first
    agent = Agent(window_size, False,"newmodel_ep1200")
    # get closing stock price
    data = getStockClosingDataVec("DJI")
    closingLength = len(data) - 1
    batch_size = 32

    state = getState(data, 0, window_size + 1)
    total_reward = 0
    agent.inventory = []

    for t in range(closingLength-5):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        lowestLow = min(data[t],data[t+1],data[t+2],data[t+3],data[t+4])

        # instead of buying or selling predict the lowest low categorically for 30min time frame
        if action == 0: # lowest low in >50 range of close

            # check how far away from lowest low
            if(abs(data[t]-lowestLow))>=50:
                print("Don't Buy: " + formatPrice(data[t]) + "no margin of safety")
                reward =25
            else:
                reward = -25

        elif action == 1: # lowest low in <25 range of close

            # check how far away from lowest low
            if(abs(data[t]-lowestLow))<25:
                print("Buy: " + formatPrice(data[t]) + "Lowest low with margin of safety")
                reward = 75
            else:
                reward = -75
        elif action == 2:
            if(abs(data[t]-lowestLow))>=25 and (abs(data[t]-lowestLow))<50:
                print("Buy: " + formatPrice(data[t]) + "Long with cautious, get ready")
            else:
                reward = -50

        total_reward+=reward

        done = True if t == closingLength - 6 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("^DJI" + " Total reward: " + formatPrice(total_reward))
            print("--------------------------------")


        if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if t % 100 == 0:
            agent.model.save("model_backup/newmodel_ep" + str(t))
    # alphaGenerator.mnistTest()





# measure and evaluate system developed
def performanceTest():
    # Calmar
    # The Calmar ratio discounts the expected excess return of a portfolio by the worst expected maximum draw down for that portfolio,

    # simulation
    # Returns from the portfolio (r) and market (m)
    returns = nrand.uniform(-1, 1, 50)
    # Expected return
    averageExpectedReturn = np.mean(returns)

    # Risk Free Rate assumption that 6% from other investment as benchmark (Opportunity cost)
    riskFreeRate = 0.06

    print("Calmar Ratio =", calmar_ratio(averageExpectedReturn, returns, riskFreeRate))

    # Monte Carlo
    pass



# using indicator to trade demo account
def automateTrading():
    # using ml model prediction and stochastic indicator to trade daily range
    pastData = getHistoricalData()
    pastDataWithIndicator = constructIndicator(pastData)


    # using past 20 day data
    # machineLearning(pastDataWithIndicator)
    machineLearning()
    print(datetime.today())
    # limit to 2 trades from # 9.30pm US market opening


'''
what is the key outcome?
1) high accuracy prediction model on the index using past data predicting (next week close or average)
2) how accurate >95% at 15 or 25 points (cater for slippages)
3) robust when back tested against historical data not
4) reinforcement learning trading algo that runs on top of prediction and raw data
5) automate trade demo using model and algo
'''
if __name__ == "__main__":
    automateTrading()
    # performanceTest()
    # after ML alpha generator is able to predict next day ohlc for the following data with high degree of accuracy in automatedtrading
    # proceed to build reinforcement learning and use performanceTest calmar ratio as fitness score



    #TODO after all is set and done final todo is to use it on CFD account