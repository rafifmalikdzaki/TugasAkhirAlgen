def test(a):
    return a

for i in range(0, 10):
    if cross := test(i):
        print(cross)
    else:
        cross = "test"
        print(cross)
    
#%%
