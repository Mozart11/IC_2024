import matplotlib.pyplot as plt
import numpy as np

samples_KNN = np.load("knn/sample_complexity_curve.npz")
samples_DT = np.load("decisionTree/sample_complexity_curve.npz")
samples_SVM = np.load("svm/sample_complexity_curve.npz")

xporcentagem = []
inc = 5
for i in range(20):
    xporcentagem.append(inc)
    inc += 5

xknn, yknn = samples_KNN["samples"], samples_KNN["train"]
xdt, ydt = samples_DT["samples"], samples_DT["train"]
xsvm, ysvm = samples_SVM["samples"], samples_SVM["train"]

xknn_std = samples_KNN["std"]
xdt_std =  samples_DT["std"]
xsvm_std =  samples_SVM["std"]

plt.plot(xknn, yknn, marker="s" ,label="KNN")          # fill_between = std # fill_between = std > Melhor tirar, não faz sentido
#plt.fill_between(xknn, yknn - xknn_std, yknn + xknn_std, alpha=0.3)
plt.plot(xdt, ydt,marker="^", label="Decision Tree")   
#plt.fill_between(xdt, ydt - xdt_std, ydt + xdt_std, alpha=0.3)
plt.plot(xsvm,ysvm, marker="o", label="SVM") 
#plt.fill_between(xsvm, ysvm - xsvm_std, ysvm + xsvm_std, alpha=0.3)
plt.legend()      
plt.grid()
plt.title("Sample Complexity Curve")
plt.xlabel("Number of Training Samples (28x28)")
plt.ylabel("Test Accuracy (%)")
plt.ylim(66,99)
plt.xlim(-5200,56000+8000)
plt.xticks(np.arange(2800,56001,2800*3))
plt.yticks(np.arange(67,100,2))
plt.show()
