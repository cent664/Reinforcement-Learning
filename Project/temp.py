a, b = 1, 2

f = open("action.txt", "a")
for i in range(0, 10):
    f.write("a = {}. b = {}\n".format(a, b))