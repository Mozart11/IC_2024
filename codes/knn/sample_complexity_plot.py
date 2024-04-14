import matplotlib.pyplot as plt
import numpy as np

samples_KNN = np.load("./sample_complexity_curve.npz")
samples_DT = np.load("../decisionTree/sample_complexity_curve.npz")
samples_SVM = np.load("../svm/sample_complexity_curve.npz")

xporcentagem = []
inc = 5
for i in range(20):
    xporcentagem.append(inc)
    inc += 5

yknn = samples_KNN["train"] 
xknn = samples_KNN["samples"]

ydt = samples_DT["train"]
xdt = samples_KNN["samples"]

ysvm = samples_SVM["train"]
xsvm = samples_SVM["samples"]

plt.plot(xknn, yknn, marker="s" ,label="KNN")          # fill_between = std
plt.plot(xdt, ydt,marker="^", label="Decision Tree")   
plt.plot(xsvm,ysvm, marker="o", label="SVM") 
plt.legend()      
plt.grid()
plt.title("Sample Complexity Curve")
plt.xlabel("Number of Training Samples")
plt.ylabel("Test Accuracy (%)")
plt.ylim(66,99)
plt.xlim(-5200,56000+8000)
plt.xticks(np.arange(2800,56001,2800*3))
plt.yticks(np.arange(67,100,2))
plt.show()
