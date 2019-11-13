import pandas as pd
import matplotlib.pyplot as plt


# To clean and aggregate hourly data to daily data
def aggregate(df):
    df.drop_duplicates(subset="date", keep='first', inplace=True)
    time = df["date"].values
    data = df["Scale_Stockmarket crash"].values
    temp_data = 0
    count = 0
    temp_data_list = []
    temp_date_list = []

    for i in range(len(time) - 1):
        if time[i][0:10] == time[i+1][0:10]:
            temp_data += data[i]
            count += 1
        else:
            count += 1
            temp_data += data[i]
            temp_data_list.append(temp_data/count)

            temp_date_list.append(time[i][0:10])

            temp_data = 0
            count = 0

    plt.plot(temp_date_list, temp_data_list)
    plt.show()


# To clean hourly data and convert to candlesticks format
def convert(df, trend, stock):
    df.drop_duplicates(subset="date", keep='first', inplace=True)
    time = df["date"].values
    data = df["Scale_Stockmarket crash"].values

    date = []
    open = []
    high = []
    low = []
    close = []
    min = 99999999999999999
    max = 0

    # Looping through the trend dataframe
    for i in range(len(time) - 1):
        # Stock Exchange opens at 9.30 am - Assuming 10 am for now
        if time[i][11:16] == "10:00":
            open.append(data[i])
        # Stock Exchange closes at 4.00 pm
        if time[i][11:16] == "16:00":
            close.append(data[i])
        # Checking for data from the same day
        if time[i][0:10] == time[i+1][0:10]:
            # Checking for low and high between 10.00 am and 4.00 pm
            if "16:00" >= time[i][11:16] >= "10:00":
                if data[i] > max:
                    max = data[i]
                if data[i] < min:
                    min = data[i]
        else:
            date.append(time[i][0:10])
            high.append(max)
            low.append(min)

            max = 0
            min = 99999999999999999

    # Creating the ohlc dataframe from open, high, low, close values just obtained
    df = pd.DataFrame(list(zip(date, open, high, low, close)), columns=['Date', 'Open', 'High', 'Low', 'Close'])
    
    df_stock = pd.read_csv('{}_train.csv'.format(stock))
    df_trend = df.copy()

    # Removing non-intersecting dates between stock and trends data
    i = 0
    j = 0
    while i < len(df_stock):
        if df_stock['Date'][i] != df_trend['Date'][j]:
            df_trend.drop([j], inplace=True)
            i -= 1
        i += 1
        j += 1

    # Saving final data file to generate images from
    df_trend.to_csv("{}_candlesticks.csv".format(trend))


if __name__ == '__main__':
    trend = "Stockmarket crash"
    stock = "S&P500"
    trends_df = pd.read_csv('{}.csv'.format(trend))
    # aggregate(trends_df)
    convert(trends_df, trend, stock)