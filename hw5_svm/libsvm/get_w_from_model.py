
from collections import defaultdict
import math 
ans = defaultdict(int)
with open("satimage_5vo.scale.model", 'r') as f:
    for line in f.readlines():
        if line.find(':') != -1: # It's a SV
            l_list = line.split("\n")[0].rstrip().split(" ")

            y_alpha = float(l_list[0])
            for element in l_list[1:]:
                idx, value = element.split(":")
                ans[int(idx)] += float(value)*y_alpha

print(f"w = {ans}")

# Get |w|
l_acc = 0
for i in ans:
    l_acc += ans[i]**2
print(math.sqrt(l_acc))
