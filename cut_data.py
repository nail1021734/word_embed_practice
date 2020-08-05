f=open('data.txt','r',encoding='utf8')
data=f.readlines()
f.close()
segment=len(data)//40
for index in range(40):
    f=open('data'+str(index)+'.txt','w',encoding='utf8')
    for i in data[segment*index:segment*(index+1)]:
        f.write(i)
    f.close()