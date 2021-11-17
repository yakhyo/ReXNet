# 1
n = int(input("N="))
nums = list(map(int, input().split()))


def func(nums):
    if len(nums) == 1:
        exit()
    for i in range(len(nums)):
        esum = sum(nums[:i + 1])
        if esum % 2 == 0:
            print("even=", esum, end=" ")
            osum = sum(nums[i + 1:])
            if osum % 2 != 0:
                print("odd=", osum)
                func(nums[i + 1:])


func(nums)


# 2

"""This is also one of my try to solve this problem


nums = [1, 13, 7, 3, 25, 11, 9]

def even_odd(nums, even=True):
    even = even
    count = 0
    if len(nums) == 1:
        exit()
    if even:
        for i in range(len(nums)):
            esum = sum(nums[:i + 1])
            if esum % 2 == 0:
                print("even=", esum, end=" ")
                osum = sum(nums[i + 1:])
                if osum % 2 != 0:
                    print("odd=", osum)
                    count += 1
                    even = False
                    even_odd(nums[i + 1:], even)
        print()
    else:
        print("-------------------------------------")
        for i in range(len(nums)):
            osum = sum(nums[:i + 1])
            if osum % 2 != 0:
                print("odd=", osum, end=" ")
                esum = sum(nums[i + 1:])
                if esum % 2 == 0:
                    print("even=", esum)
                    count += 1
                    even = True
                    even_odd(nums[i + 1:], even)
        print()


even_odd(nums)

"""
