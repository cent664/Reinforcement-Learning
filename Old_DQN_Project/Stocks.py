from selenium import webdriver
import time
from Rename import rename_file
import random


def scraping(stockname):
    prefs = {
        "download.default_directory": r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\Data",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True
    }

    # example option: add 'incognito' command line arg to options
    option = webdriver.ChromeOptions()
    option.add_experimental_option('prefs', prefs)
    option.add_argument("--incognito")
    option.add_argument("--start-maximized")

    # Create new instance of chrome in incognito mode
    browser = webdriver.Chrome(executable_path=r"C:\Users\Flann lab\Downloads\chromedriver_win32\chromedriver.exe", chrome_options=option)

    # Go to website of interest
    browser.get("https://finance.yahoo.com/")

    search_bar = browser.find_element_by_id("yfin-usr-qry")  # search bar
    search_bar.send_keys(stockname)  # typing in
    time.sleep(random.randint(10, 15))  # In case of time outs

    search_button = browser.find_element_by_id("header-desktop-search-button")  # finding the search button
    search_button.click()  # clicking on search
    time.sleep(random.randint(10, 15))  # In case of time outs

    hist_data = browser.find_element_by_xpath('//span[text()="Historical Data"]')  # finding historical data
    hist_data.click()  # clicking on historical data
    time.sleep(random.randint(10, 15))  # In case of time outs

    download_button = browser.find_element_by_xpath('//span[text()="Download Data"]')  # finding download button
    download_button.click()  # clicking on download
    time.sleep(random.randint(10, 15))  # In case of time outs


if __name__ == '__main__':
    stockname = 'S&P500'
    scraping(stockname)

    input_dir = r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\Data"
    target = r"C:\Users\Flann lab\PycharmProjects\Reinforcement-Learning\{}_Stock.csv".format(stockname)
    rename_file(input_dir, target)