import numpy as np
def dedup(path):
    back = []
    with open(path) as f:
        line = f.readline()
        while line:
            line = line.replace('\n','')
            l1 = line.replace('.', '').split(" ")
            l2 = list(set(l1))
            l2.sort(key=l1.index)
            index = []
            for i in range(len(l2)):
                if l2[i] in back:
                    index.append(i)
            l2 = np.array(l2)
            l2 = np.delete(l2,index)
            l2 = list(l2)
            back = back+l2
            line = f.readline()
        return back
print(dedup("hhh.txt"))