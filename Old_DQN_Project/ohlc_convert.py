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
def convert(df, trend, stock, mode):
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
    df = pd.DataFrame(list(zip(date, open, high, low, close)), columns=['Date', 'Open', 'High', 'Low', 'Close'])

    df_stock = pd.read_csv('{}_{}.csv'.format(stock, mode))
    df_trend = df.copy()

    df_trend.to_csv("{}_{}_candlesticks.csv".format(trend, mode))

    # Removing non-intersecting dates between stock and trends data
    i = 0
    j = 0
    while i < len(df_stock):
        print(i, j)
        if df_stock['Date'][i] < df_trend['Date'][j]:
            df_stock.drop([i], inplace=True)
            j -= 1
        if df_stock['Date'][i] > df_trend['Date'][j]:
            df_trend.drop([j], inplace=True)
            i -= 1
        i += 1
        j += 1

    # Saving final data file to generate images from
    df_trend.to_csv("{}_{}_candlesticks.csv".format(trend, mode))


if __name__ == '__main__':
    trend = "Stockmarket crash"
    stock = "S&P500"
    mode = "Test"
    trends_df = pd.read_csv('{}_{}.csv'.format(trend, mode))
    # aggregate(trends_df)
    convert(trends_df, trend, stock, mode)