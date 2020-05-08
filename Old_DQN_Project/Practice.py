import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test():
    print(ord('A'))

    for i in range(9, -1, -1):
        print(i)



def list_comprehension():
    # List comprehension - Random, zip
    for i in range(10):
        print(random.random())  # Random float between [0, 1)

    test_ar1 = []
    test_ar2 = []

    # Random seed for both arrays
    for i in range(10):
        test_ar1.append(random.randint(0, 10))
        test_ar2.append(random.randint(0, 10))

    print(test_ar1, test_ar2)

    test_ar = [(x + y) for (x, y) in zip(test_ar1, test_ar2)]
    print(test_ar)

    a = []
    a += [3] * 2
    a += [5] * 6
    a.append([4] * 4)
    print(a)

    # Deleting
    temp = [0, 1, 2, 3, 4, 0, 0, 6, 7, 8]
    l = len(temp)
    i = 0
    count = 0
    while i < l:
        if temp[i] == 0:
            del temp[i]
            count += 1
            l -= 1
            i -= 1
        i += 1

    temp += [0]*count
    print(count)
    print(temp)

    temp = [1, 3, 2]
    print(np.unique(temp))


def string_manipulation():
    # String manipulation - Splicing, finding
    # try, catch (except), finally

    # Conversion, try, except
    s = "fatfat664"
    for i in range(len(s)):
        try:
            if type(int(s[i])) == int:
                print('{} is an integer!'.format(s[i]))
        except:
            print('Error! is not an integer! DansGame'.format(s[i]))
        # finally:
        #     print('I think finally is irrelevant')
        else:
            print('Nothing went wrong!')

    # Counting
    char = 'f'
    print("Number of {}s in {} = {}".format(char, s, s.count(char)))

    # Splicing
    print(s[0:3])
    print(s[6:len(s) + 1])

    # Finding - First occurrence
    letter = 'f'
    if s.find(letter) == -1:
        print('{} was not found'.format(letter))
    else:
        print('{} was found at position {}!'.format(letter, s.find(letter)))

    # Replace
    s = s.replace('fatfat', 'cent')
    print(sorted(s))


def zeros_and_ones():
    one_d = np.zeros(10)
    print(one_d.shape)

    one_d = np.ones(10)
    print(one_d * 5)

    two_d = np.zeros([5, 10])  # Rows, Columns
    print(two_d)

    two_d = 2 * np.ones([5, 10])  # Rows, Columns
    print(two_d * 5)


def arrange_reshape_concat():
    # np arrays - simple 1D with ones/zeros
    a = np.zeros(5)
    b = 2 * np.ones(5)

    # Important to reshape to make it 2D, to be able to concat along vertical axis (1 = more columns)
    a = np.reshape(a, (1, 5))
    b = np.reshape(b, (1, 5))

    c = np.concatenate((a, b), axis=1)  # 0 = More rows, 1 = More columns
    print(c)

    # np arrays - complex 2D reshape with zeros/ones
    a = np.zeros((2, 2))
    b = np.ones((2, 2))
    print(a.shape, b.shape)

    a = np.reshape(a, (1, 4))  # (Rows, Columns)
    b = np.reshape(b, (1, 4))

    print(a.shape, b.shape)
    c = np.concatenate((a, b), axis=1)
    print(c)

    # np arrays - complex 2D reshape with arrange
    a = np.arange(0, 4, 1)
    b = np.arange(0, 4, 2)

    a = np.reshape(a, (2, 2))
    b = np.reshape(b, (2, 1))
    c = np.concatenate((a, b), axis=1)
    print(c)

    # lists - concatenation and duplication
    a = [1, 2, 3, 4, 5]
    b = [10, 213, 12, 323, 2]
    print(a + b)
    print(3 * a)

    c = [a, b]
    c = np.reshape(c, (1, -1))
    print(c[0])

    # def slicing():
    s = ["f", "a", "t", "f", "a", "t", "6", "6", "4"]

    print(s[len(s):0:-1])
    print(s[len(s)::-1])
    s.reverse()
    print(s)


def dictionary_set():
    # Initializing
    test_dict = {1: "one",
                 2: "two",
                 3: "three"}

    # Looping
    for i in test_dict:
        print(i, test_dict[i])
    # OR
    for i, j in test_dict.items():
        print(i, j)

    # print(test_dict.clear())
    # print(test_dict.items())
    # print(test_dict.keys())
    # print(test_dict.values())
    # print(test_dict.popitem())  # Last item
    # print(test_dict.pop(1))  # Specific key
    # del test_dict[1]  # Alternative to pop
    # del test_dict  # Delete the whole dict

    # Appending
    test_dict[4] = "four"

    print(test_dict)
    print("Length = {}".format(len(test_dict)))

    # Searching
    if 4 in test_dict:
        print("Bingo! Popping now")
        test_dict.pop(4)
    print(test_dict)
    print("New length = {}".format(len(test_dict)))

    print(test_dict.popitem())

    # Sets
    sett = {1}  # 1 -- {} only is dict, {1:1} is dict
    sett = set()  # 2
    set1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    set2 = {6, 7, 8, 9}

    set1.add(10)
    print(set1.intersection(set2))

def xor():
    a = 5
    b = 8

    print(a ^ b)
    print(b ^ a)
    print(a ^ (a ^ b))
    print(b ^ (a ^ b))

if __name__ == '__main__':
    # test()
    # list_comprehension()
    # string_manipulation()
    # zeros_and_ones()
    # arrange_reshape_concat()
    # slicing()
    # dictionary_set()
    xor()