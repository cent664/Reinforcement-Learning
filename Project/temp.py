from makeshift_env import StockTradingEnv
# DQN Stocks

env = StockTradingEnv()  # Object of the environment

a = env.reset('test')
i1 = a[0]

b = env.reset('train')
i2 = b[0]

if (i1 == i2).all():
    print(True)