import numpy as np

a =np.random.randint(1,50,size= (5,4))
print(a)
print(a.dtype)

for x in range(0,5):
    for y in range (0,4):
        if x + y == 3:
            print(a[x][y])
print(" ")
for x in range(0,5):
    max = a[x][y]
    for y in range (0,4):
        if a[x][y] > max:
            max = a[x][y]
    print(max)
print(" ")

b = np.array([])

c = np.mean(a)
for x in range(0,5):
    for y in range (0,4):
        if a[x][y] < c:
            b = np.append(b,a[x][y])
b = b.astype('int64')
print(b)

print(" ")

def numpy_boundary_traversal(arr):
    rows,col = arr.shape
    for x in range(col):
        print(arr[0][x])
    for x in range(1,rows):
        print(arr[x][col -1])
    for x in range(col -2, -1, -1):
        print(arr[rows-1][x])
    for x in range(rows - 2, 0 ,-1):
        print(arr[x][0])
    
numpy_boundary_traversal(a)