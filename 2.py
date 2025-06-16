import numpy as np

a = np.random.rand(20)
b = np.random.randint(0,9,20)

c = a + b
rounded_c = np.round(c,2)
print(c)
print(rounded_c)
print(np.max(rounded_c))
print(np.min(rounded_c))
print(np.median(rounded_c))

for x in range(20):
    if rounded_c[x] < 5:
        rounded_c[x] = rounded_c[x]*rounded_c[x]
rounded_c = np.round(rounded_c,2)
print(rounded_c)

def swap(arr,a,b):
    temp = arr[b]
    arr[b] = arr[a]
    arr[a] = temp
    return

def numpy_alternate_sort(arr):
    for x in range(10):
        print(x)

numpy_alternate_sort(rounded_c)