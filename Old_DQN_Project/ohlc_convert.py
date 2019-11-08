import pandas as pd
import matplotlib.pyplot as plt


# To aggregate and clean hourly data to daily data
def aggregate(df):
    df.drop_duplicates(subset="date", keep='first', inplace=True)
    time = df["date"].values
    data = df["Scale_['Stockmarket crash']"].values
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


# To clean hourly data and convert to candlesticks
def convert(df, keyword):
    df.drop_duplicates(subset="date", keep='first', inplace=True)
    time = df["date"].values
    data = df["Scale_['Stockmarket crash']"].values

    date = []
    open = []
    high = []
    low = []
    close = []
    min = 99999999999999999
    max = 0

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
            min = 0

    df = pd.DataFrame(list(zip(date, open, high, low, close)), columns=['Date', 'Open', 'High', 'Low', 'Close'])
    df.to_csv("{}_Trends_candlesticks.csv".format(keyword))

if __name__ == '__main__':
    keyword = "Stockmarket crash"
    trends_df = pd.read_csv('{}_Trends.csv'.format(keyword))
    # aggregate(trends_df)
    convert(trends_df, keyword)