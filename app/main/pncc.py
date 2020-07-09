def charu(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        while i>=1 and key < arr[i-1]:
            arr[i] = arr[i-1]
            i-=1
        arr[i] = key


def part(arr,low,high):
    pr = arr[low]
    while low<high:
        while low<high and arr[high]>=pr:
            high -=1
        arr[low] = arr[high]
        while low < high and arr[low] <= pr:
            low +=1
        arr[high] = arr[low]
    arr[low] = pr
    return low


def quick(arr,low,high):
    if low < high:
        pi = part(arr,low,high)
        quick(arr,low,pi-1)
        quick(arr,pi+1,high)
    return arr

def sec(arr):
    for i in range(len(arr)):
        mid = i
        for j in range(i+1,len(arr)):
            if arr[mid] > arr[j]:
                mid = j
        arr[i],arr[mid] = arr[mid],arr[i]

def merge_sort(arr):
    if len(arr)<=1:
        return arr
    mid = int(len(arr)/2)
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left,right)

def merge(left,right):
    c = []
    i = j =0
    while i<len(left) and j < len(right):
        if left[i]<right[j]:
            c.append(left[i])
            i+=1
        if left[i]>=right[j]:
            c.append(right[j])
            j+=1
        if i == len(left):
            for m in right[j:]:
                c.append(m)
        else:
            for n in left[i:]:
                c.append(n)
    return c

def pao(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]

def quan(arr,low,high):
    if low>=high:
        print(arr)
    else:
        low1 = low
        for i in range(low,high):
            arr[i],arr[low1] = arr[low1],arr[i]
            quan(arr,low+1,high)
            arr[i], arr[low1] = arr[low1], arr[i]

# arr = [1,2,3]
# quan(arr,0,len(arr))

def long_test(arr):
    c = ""
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i:len(arr)-j] in arr[i+1:]:
                if len(c)<=len(arr[i:len(arr)-j]):
                    c = arr[i:len(arr)-j]
    return c


arr = "1.2.3.4"
print(arr.split("."))