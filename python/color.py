
global f_dict,g_dict
f_dict={}
g_dict=[]
def f(alphas,n,k):
    global f_dict,g_dict  
    if tuple(alphas) in f_dict:
       return f_dict[tuple(alphas)]
    if n==1:
        return 1
    if n==0:
        return 0
    if k==1:
        return 0
    res=0
    for i in range(len(alphas)):
        if alphas[i]==0:
            continue
        val = g(alphas,i,n,k)
        res += val
    #f_dict[tuple(alphas)] = res
    return res
def g(alphas,k,n,num):
    global g_dict    
    if tuple(alphas) in g_dict[k]:
        return g_dict[k][tuple(alphas)]
    if n==1:
        if alphas[k]==1:
            return 1
        else:
            return 0
    if alphas[k]==0:
        return 0
    if alphas[k]==1:
        num_ = num-1
    else:
        num_ = num
    alphas_ = alphas.copy()
    alphas_[k]-= 1
    g_val = g(alphas_,k,n-1,num_)    
    f_val = f(alphas_,n-1,num_)
    val = f_val-g_val
    g_dict[k][tuple(alphas)] = val%1000000007 
    return val
k = int(input())
g_dict =[{}for i in range(k)]
alphas = input()
alphas = alphas.split()
for i in range(len(alphas)):
    alphas[i] = int(alphas[i])
n = sum(alphas)
print(f(alphas,n,k)%1000000007)