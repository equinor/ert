import random

with open("digit-blob", "w") as f:
    for i in range(1024 * 1024 * 1024):
        f.write(str(i % 10))
    f.write("\n")
