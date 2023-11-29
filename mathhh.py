import math
T = 10000
c = 1000
gamma = 0.5772
R = T / (2 * c) + 2 - math.log(T/c) - gamma
print(R)

hs = 0
for i in range(1, T//c):
    hs += 1/(i+2)

R2 = T/(2 * c) + 1/2 - hs
print(R2)

R3 = 0
for i in range (1, T//c):
    R3 += (i + 1)/(i+2) - 1/2
print(R3)

R4 = 0
for i in range (1, T//c):
    R4 += (i + 2)/(i+2) - 1/2 - 1/(i+2)
print(R4)

R5 = 0
for i in range (1, T//c):
    R5 += 1/2 - 1/(i+2)
print(R5)