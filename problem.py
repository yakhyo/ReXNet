
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
