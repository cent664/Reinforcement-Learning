import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# To clean and aggregate hourly data to daily data
def aggregate(df):
    df.drop_duplicates(subset="date", keep='first', inplace=True)
    time = df["date"].values
    data = df["Scale_{}".format(trend)].values
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


# To clean hourly data, aggregate to daily data and convert to candlesticks format
def convert_and_clean(df_stock, df_trend, trend, stock, final_length):
    df_trend.drop_duplicates(subset="date", keep='first', inplace=True)
    time = df_trend["date"].values
    data = df_trend["Scale_{}".format(trend)].values

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
        date_temp, time_temp = time[i].split()
        dateplusone_temp, _ = time[i+1].split()

        if time_temp == "10:00:00":
            open.append(data[i])
        # Stock Exchange closes at 4.00 pm
        if time_temp == "16:00:00":
            close.append(data[i])
        # Checking for data from the same day
        if date_temp == dateplusone_temp:
            # Checking for low and high between 10.00 am and 4.00 pm
            if "16:00:00" >= time_temp >= "10:00:00":
                if data[i] > max:
                    max = data[i]
                if data[i] < min:
                    min = data[i]
        else:
            date.append(date_temp)
            high.append(max)
            low.append(min)

            max = 0
            min = 99999999999999999

    # Creating the ohlc dataframe from open, high, low, close values just obtained
    df_trend = pd.DataFrame(list(zip(date, open, high, low, close)), columns=['Date', 'Open', 'High', 'Low', 'Close'])

    # To account for holidays
    if datetime.strptime(df_stock['Date'].iloc[len(df_stock) - 1], '%Y-%m-%d') != datetime.strptime(df_trend['Date'].iloc[len(df_trend) - 1], '%Y-%m-%d'):
        return False
    else:

        i = 0
        j = 0
        ls = len(df_stock)

        # Removing non-intersecting dates between stock and trends data
        while i < ls:
            if datetime.strptime(df_stock['Date'].iloc[i], '%Y-%m-%d') == datetime.strptime(df_trend['Date'].iloc[j], '%Y-%m-%d'):
                # print(df_stock['Date'].iloc[i], df_trend['Date'].iloc[j], "stock = trend")
                i += 1
                j += 1
            # Deleting from stocks
            elif datetime.strptime(df_stock['Date'].iloc[i], '%Y-%m-%d') < datetime.strptime(df_trend['Date'].iloc[j], '%Y-%m-%d'):
                # print(df_stock['Date'].iloc[i], df_trend['Date'].iloc[j], "stock < trend")
                df_stock.drop(df_stock.index[i], inplace=True)
                ls -= 1
            # Deleting from trends
            elif datetime.strptime(df_stock['Date'].iloc[i], '%Y-%m-%d') > datetime.strptime(df_trend['Date'].iloc[j], '%Y-%m-%d'):
                # print(df_stock['Date'].iloc[i], df_trend['Date'].iloc[j], "stock > trend")
                df_trend.drop(df_trend.index[j], inplace=True)

        # Taking the last 'final_length' number of data points
        df_stock = df_stock[len(df_stock) - final_length:]
        df_trend = df_trend[len(df_trend) - final_length:]

        # Saving final data files to generate images from
        df_stock.to_csv("{}_Stock.csv".format(stock))
        df_trend.to_csv("{}_Trend.csv".format(trend))

        return True


if __name__ == '__main__':
    trend = "S&P500"
    stock = "S&P500"
    final_length = 217
    stock_df = pd.read_csv('{}_Stock.csv'.format(stock))
    trend_df = pd.read_csv('{}_Trend.csv'.format(trend))
    convert_and_clean(stock_df, trend_df, trend, stock, final_length)
    # aggregate(trends_df)