
# Transform data into '5' versus 'not 5'
# 
s = ""
with open('libsvm/satimage.scale.t', 'r') as f:
    for line in f.readlines():
        if line[0] != '6':
            s += '1' + line[1:] # "not X"
        else:
            s += '2' + line[1:] # "is X"

with open('libsvm/satimage_6vo.scale.t', 'w') as f:
    f.write(s)



