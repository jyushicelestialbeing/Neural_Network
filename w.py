import random


filename = 'sample.txt'
temp = []
j = 0
if __name__ == '__main__':
    print('start')
    for i in range(100):
        a = random.randint(1,100)
        b = random.randint(1,100)
        if a >=60 and b>=60:
            j = 1
        else:
            j = 0
        temp.append(str(a)+','+str(b)+':'+str(j)+'\n')
    with open(filename,'w') as f:
        for i in temp:
            f.writelines(i)
    print('finish')
