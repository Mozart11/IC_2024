import matplotlib.pyplot as plt
import numpy as np

samples = np.load("./sample_complexity_curve.npz")
x = []
inc = 5
for i in range(20):
    x.append(inc)
    inc += 5

y = samples["train"] 
x2 = samples["samples"]
print(x2)
plt.scatter(x2, y)
plt.plot(x2, y)          # fill_between = std
plt.grid()
plt.title("Sample Complexity Curve")
plt.xlabel("Number of Training Samples")
plt.ylabel("Test Accuracy (%)")
plt.ylim(85,99)
plt.xlim(2800,56000)
plt.xticks(np.arange(2800,56001,5320))
plt.yticks(np.arange(85,100,1))
plt.show()
