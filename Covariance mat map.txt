

import sys

import numpy as np
import pandas as pd
from numpy import matrix
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import matrix
names = []
n = int(input("total number of files: ")) #input total number of matrices present in dir
l = [d for d in range(1,n+1)]
for d in range(0,n):
    x = 'k20dis'+str(l[d])+'.xlsx'
    names.append(x)
print(names)
mainarray = [[]]
resArrb = np.zeros((31, 31))
arraytemp = resArrb
for d in range(0,n):
    #Reading excel csv datafile
    df = pd.read_excel((names[d]),header=None)

    resArr = np.zeros((31, 31))

    arr = df
    arr= np.array(arr)
    for i in range(len(arr)):
        for k in range(i,len(arr)):
            temp =0
            for j in range(3):

                temp = temp + (arr[k][j]-arr[i][j])**2
                resArr[i][k] = temp ** .5
                resArr[k][i]= temp**.5






    df=resArr
    resArr = np.array(resArr)
    Y = np.cov(resArr,bias=True)

    plt.imshow(Y, alpha=10, cmap='seismic')
    plt.savefig('cov20k ' + str(l[d]) + '.jpg')



