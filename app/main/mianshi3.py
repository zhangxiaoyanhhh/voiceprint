def ploy(arr, x):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]*pow(x,i)
    return sum


def jie(a):
    s = 1
    for j in range(1,a+1):
        s = s*j
    return s


def poly_taylor(p, r):
    back = []
    for j in range(len(p)):

        if j == 0:
            back.append(ploy(p, r))
        else:
            # print(ploy(p, r))
            p.pop(0)
            # print(p)
            p1 = []
            for n in range(1,len(p)+1):
                p1.append(p[n-1]*(n))
            # print(p1)
            p = p1
            back.append(ploy(p, r)/jie(j))
    return back

print(poly_taylor([2,-5,7,-4,1],3))