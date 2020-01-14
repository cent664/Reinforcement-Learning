# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

def binarySearch(l, r):
    while l <= r:
        mid = int((l + r) / 2)

        if isBadVersion(mid) == True and isBadVersion(mid - 1) == False:
            return mid
        elif isBadVersion(mid) == True:
            r = int(mid - 1)
        elif isBadVersion(mid) == False:
            l = int(mid + 1)
    return -1


class Solution:
    def firstBadVersion(self, n):
        pos = binarySearch(1, n)
        if pos == -1:
            return n
        else:
            return pos