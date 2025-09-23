# This is a sample Python script.
import math
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random
from collections import deque
from operator import mod


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')
    # x = 0  # Press ⌘F8 to toggle the breakpoint.
    # while x < 100:
    #     print(x)
    print('*' * 50)
        # x += 1


def two_sum(nums, target):
    visitedMap = {}
    for (index, number) in enumerate(nums):
        if abs(target - number) in visitedMap:
            return [visitedMap[target - number], index]
        else:
            visitedMap[number] = index

    return [-1, -1]


def find_max_k(nums):
    numbersSet = set()
    maxSoFar = -1

    for number in nums:
        if numbersSet.__contains__(-1 * number):
            if abs(number) >= abs(maxSoFar):
                maxSoFar = abs(number)
        numbersSet.add(number)
    return maxSoFar


def roman_to_int(s):
    romanToIntMap = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
    }
    result = 0
    index = 0
    while index < len(s):
        if ((index + 1) < len(s)) and (romanToIntMap.get(s[index]) < romanToIntMap.get(s[index + 1])):
            result = result + romanToIntMap.get(s[index + 1]) - romanToIntMap.get(s[index])
            index += 2
        else:
            result = result + romanToIntMap.get(s[index])
            index += 1

    return result


def is_valid_parantheses(s):
    parenthesesMap = {
        ")": "(",
        "}": "{",
        "]": "["
    }
    stack = deque()

    index = 0
    while index < len(s):
        if s[index] in parenthesesMap:
            if stack.maxlen == 0:
                return False
            else:
                if stack.pop() != parenthesesMap.get(s[index]):
                    return False
        else:
            stack.append(s[index])
        index += 1
    return True


def add_binary(a, b):
    map = {
        2: 10,
        3: 11
    }

    carry = '0'
    indexA = len(a) - 1
    indexB = len(b) - 1
    result = ''

    while indexA >= 0 and indexB >= 0:
        sum = int(a[indexA]) + int(b[indexB]) + int(carry)
        if sum in map:
            carry = "1"
            sum = map[sum]
            sum = sum % 10
        else:
            carry = "0"
        result = str(sum) + result
        indexA -= 1
        indexB -= 1

    while indexA >= 0:
        sum = int(a[indexA]) + int(carry)
        if sum in map:
            carry = "1"
            sum = map[sum]
            sum = sum % 10
        else:
            carry = "0"
        result = str(sum) + result
        indexA -= 1

    while indexB >= 0:
        sum = int(b[indexB]) + int(carry)
        if sum in map:
            carry = "1"
            sum = map[sum]
            sum = sum % 10
        else:
            carry = "0"
        result = str(sum) + result
        indexB -= 1

    if carry != '0':
        result = str(carry) + result

    return result


def climb_stairs(n):
    x = 0
    memoization = [0]*n
    greedy(x, n, memoization)
    return memoization[0]


def greedy(x, n, memoization):
    if x == n:
        return 1

    if x > n:
        return 0

    if memoization[x] != 0:
        return memoization[x]

    memoization[x] = greedy(x + 1, n, memoization) + greedy(x + 2, n, memoization)
    return memoization[x]


def merge_lists(nums1, m, nums2, n):
    index = m + n - 1
    m -= 1
    n -= 1
    while m >= 0 and n >= 0:
        while m>=0 and n>=0 and nums1[m] >= nums2[n]:
            nums1[index] = nums1[m]
            index -= 1
            m -= 1
        while m>=0 and n>=0 and nums2[n] > nums1[m]:
            nums1[index] = nums2[n]
            index -= 1
            n -= 1
        print(m, n, index)

    while m >= 0:
        nums1[index] = nums1[m]
        index -= 1
        m -= 1

    while n >= 0:
        nums1[index] = nums2[n]
        index -= 1
        n -= 1

    return nums1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('')

    # print(two_sum([2, 7, 11, 15], 9))

    # print(find_max_k([-10, 11, -11, 12, 13, -15, 7, -7]))

    # print(roman_to_int("MCMXCIV"))
    # print(roman_to_int("III"))

    # print(is_valid_parantheses("[{()}]"))

    # print(add_binary("1101", "101"))

    # print(climb_stairs(5))
    # print(climb_stairs(3))
    # print(climb_stairs(2))
    # print(climb_stairs(38))


    # print(merge_lists([1,2,3,0,0,0], 3, [2,5,6], 3))

    # print(inorderTraversal([1, null, 2, 3]))

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
# BA 606 73 - Sem: Exp Lrng
# BA 625 71 - Negotiation Mgt
# BA 500 86- Sem: Exp Lrng
