def ploy(arr, x):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]*pow(x,i)
    return sum
print(ploy([1,2,3],2))
