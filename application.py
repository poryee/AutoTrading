__author__ = 'po'


from ig_service import IGService
from ig_service_config import * # defines username, password, api_key, acc_type,
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#%matplotlib inline



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

    #Price resolution (SECOND, MINUTE, MINUTE_2, MINUTE_3, MINUTE_5, MINUTE_10, MINUTE_15, MINUTE_30, HOUR, HOUR_2, HOUR_3, HOUR_4, DAY, WEEK, MONTH)
    resolution = 'DAY' # resolution = 'H', '1Min'



    #(yyyy:MM:dd-HH:mm:ss)
    today = datetime.today()
    startDate = str(today.date() - timedelta(days=30)).replace("-",":") + "-00:00:00"
    endDate = str(today.date() - timedelta(days=0)).replace("-",":") + "-00:00:00"

    response = ig_service.fetch_historical_prices_by_epic_and_date_range(epic,resolution,startDate,endDate)
    return response['prices']




def getAverage(dataArray):
    tempList = []
    for priceObject in dataArray:
        tempList.append((priceObject['ask']+priceObject['bid'])/2)
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
    pastData['%K'] = ((pastData['averageClose'] - pastData['lowestLow']) / (pastData['highestHigh'] - pastData['lowestLow']))*100

    #Create the "%D" column in the DataFrame moving average of calculated K
    pastData['%D'] = pastData['%K'].rolling(window=3).mean()

    # drop 14 bar ago
    pastData.drop(pastData.index[:15], inplace=True)

    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,10))

    pastData['averageClose'].plot(ax=axes[0]); axes[0].set_title('Close')
    pastData[['%K','%D']].plot(ax=axes[1]); axes[1].set_title('Oscillator')
    plt.show()





# using indicator to trade demo account
def automateTrading():
    pass

# measure and evaluate system developed
def performanceTest():
    pass

# proceed to use machine learning algorithm to predict the weekly price range to provide more confidence
def machineLearning():
    pass

if __name__ == "__main__":
    pastData=getHistoricalData()
    #print(pastData)
    constructIndicator(pastData)
    #TODO after all is set and done final todo is to use it on CFD account