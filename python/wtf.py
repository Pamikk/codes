n = int(input())
a = [0]*n
b = [0]*n
c = [0]*n
for i in range(n):
    x,y = input().split()
    a[i] = int(x)
    b[i] = int(y)
    for j in range(i+1):
        c[j] += a[i]*b[i-j]
for val in c:
    print(val)

