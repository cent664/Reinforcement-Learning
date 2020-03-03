import os


def rename_file(input_dir, target):
    try:
        os.remove(target)
        print("Updating previous stock data with new one.")
    except:
        print("No previous files exist. Fresh start.")

    if not os.path.exists(input_dir):
        raise ValueError("{} does not exist".format(input_dir))

    source = os.listdir(input_dir)[0]
    source = os.path.join(input_dir, source)

    if ".csv" not in source:
        raise TypeError(".csv not found")
    os.rename(source, target)


if __name__ == '__main__':
    stockname = "S&P500"
    input_dir = r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\Data"
    target = r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\{}_Stock.csv".format(stockname)
    rename_file(input_dir, target)
