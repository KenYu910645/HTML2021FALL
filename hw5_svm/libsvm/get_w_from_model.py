
from collections import defaultdict
import math 

ans = defaultdict(int)
sum_alpha = 0

with open("satimage_5vo.scale.model", 'r') as f:
    for line in f.readlines():
        if line.find(':') != -1: # It's a SV
            l_list = line.split("\n")[0].rstrip().split(" ")

            y_alpha = float(l_list[0])
            sum_alpha += abs(y_alpha)
            for element in l_list[1:]:
                idx, value = element.split(":")
                ans[int(idx)] += float(value)*y_alpha

# Get |w|
l_acc = 0
for i in ans:
    l_acc += ans[i]**2
w_len = math.sqrt(l_acc)
print(f"|w| = {w_len}")

print(f"(3) = {1/w_len}")
print(f"(4) = {sum_alpha**(-0.5)}")
print(f"(6) = {(2*sum_alpha - w_len**2)**(-0.5)}")
print(f"(7) = {(-0.5*(w_len**2) + sum_alpha)}")