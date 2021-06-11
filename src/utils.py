import itertools
import math as m

def jpeg_coefficient_coding_categories(x):
    if x == 0:
        return [('0')]
    else:
        category = len(bin(x).partition('b')[-1])
        return list(itertools.product(['0', '1'], repeat=category))

x = -9
l = jpeg_coefficient_coding_categories(x)
print(l)

mid = len(l)//2
if x < 0:
    val = l[:mid][(x % mid) - 1]
else:
    val = l[mid:][x - mid]

print("".join(val))