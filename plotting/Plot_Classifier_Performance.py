import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read in classification accuracy results
f1 = pickle.load( open( "../results/accuracy_CNN_high_compression.dat", "rb" ) )
f2 = pickle.load( open( "../results/accuracy_CNN_high_compression_adv.dat", "rb" ) )

snrs=list(np.arange(-20,18,2))
acc_cnn_high_compress=list(map(lambda x: f1[x], snrs))
acc_cnn_high_compress_adv=list(map(lambda x: f2[x], snrs))

plt.figure(),
plt.plot(snrs,acc_cnn_high_compress,linewidth=8,label='CNN high compression')
plt.plot(snrs,acc_cnn_high_compress_adv,linewidth=8,label='CNN high compression under FGSM')
plt.legend(fontsize=30)
plt.grid()
plt.xlabel("Signal to Noise Ratio",fontsize=40)
plt.ylabel("Classification Accuracy",fontsize=40)
plt.tick_params(axis='both',labelsize=30)
plt.title("Classification Accuracy on RadioML",fontsize=50)
plt.show()
