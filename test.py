class Bob:
    count = 0

    def __init__(self, value):
        Bob.count += 1
        self.value = value

    def __del__(self):
        Bob.count -= 1


l = []
for i in [1, 2, 11, 20]:
    a = Bob(i)
    l.append(a)


for element in l:
    print(element.value)

for idx in reversed(range(len(l))):
    if l[idx].value < 10:
        del l[idx]

for element in l:
    print(element.value)
print(len(l))
print(Bob.count)
