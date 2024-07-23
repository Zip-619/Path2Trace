# for line in open('tmp.txt'):
#     arr = line.strip().split(' ')
#     print('\t'.join(([a for a in arr if len(a) >0])))


# for line in open('tmp.txt'):
#     arr = line.strip().split('	')
#     print('&'.join(arr)+'\cr')

# arr = list()
# for line in open('tmp.txt'):
#     arr.append(line.strip())
#     (line.strip().split('	'))
# print(','.join(arr))


for line in open('tmp.txt'):
    arr = line.strip().split('	')
    print(arr[0] + '=[' + ','.join(arr[1:]) + ']')
