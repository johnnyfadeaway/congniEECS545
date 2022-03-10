from mfcc_compress import mfcc
import numpy as np
import matplotlib.pyplot as plt

subs_learn_data = np.load("subs_learn_data.npz")
genre_1_mat = subs_learn_data["genre_1_mat"]
genre_2_mat = subs_learn_data["genre_2_mat"]
genre_3_mat = subs_learn_data["genre_3_mat"]
genre_4_mat = subs_learn_data["genre_4_mat"]
genre_5_mat = subs_learn_data["genre_5_mat"]
genre_6_mat = subs_learn_data["genre_6_mat"]
genre_7_mat = subs_learn_data["genre_7_mat"]

genre_1_train = genre_1_mat[:,:,1:61]
genre_1_test = genre_1_mat[:,:,61:]
genre_2_train = genre_2_mat[:,:,1:61]
genre_2_test = genre_2_mat[:,:,61:]
genre_3_train = genre_3_mat[:,:,1:61]
genre_3_test = genre_3_mat[:,:,61:]
genre_4_train = genre_4_mat[:,:,1:61]
genre_4_test = genre_4_mat[:,:,61:]
genre_5_train = genre_5_mat[:,:,1:61]
genre_5_test = genre_5_mat[:,:,61:]
genre_6_train = genre_6_mat[:,:,1:61]
genre_6_test = genre_6_mat[:,:,61:]
genre_7_train = genre_7_mat[:,:,1:61]
genre_7_test = genre_7_mat[:,:,61:]

genre_1_train = genre_1_train.reshape(512*512,-1)
genre_1_test = genre_1_test.reshape(512*512,-1)
genre_2_train = genre_2_train.reshape(512*512,-1)
genre_2_test = genre_2_test.reshape(512*512,-1)
genre_3_train = genre_3_train.reshape(512*512,-1)
genre_3_test = genre_3_test.reshape(512*512,-1)
genre_4_train = genre_4_train.reshape(512*512,-1)
genre_4_test = genre_4_test.reshape(512*512,-1)
genre_5_train = genre_5_train.reshape(512*512,-1)
genre_5_test = genre_5_test.reshape(512*512,-1)
genre_6_train = genre_6_train.reshape(512*512,-1)
genre_6_test = genre_6_test.reshape(512*512,-1)
genre_7_train = genre_7_train.reshape(512*512,-1)
genre_7_test = genre_7_test.reshape(512*512,-1)

a = mfcc(genre_1_train[:,0], samplerate=8000,winstep=0.01, winlen=0.025, numcep=16)
M,N = a.shape
print(M*N)
genre_1_train_mfcc = np.empty((M*N,60))
for i in range(60):
      genre_1_train_mfcc[:,i] = mfcc(genre_1_train[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_1_test_mfcc = np.empty((M*N,37))
for i in range(37):
      genre_1_test_mfcc[:,i] = mfcc(genre_1_test[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_2_train_mfcc = np.empty((M*N,60))
for i in range(60):
      genre_2_train_mfcc[:,i] = mfcc(genre_2_train[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_2_test_mfcc = np.empty((M*N,37))
for i in range(37):
      genre_2_test_mfcc[:,i] = mfcc(genre_2_test[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_3_train_mfcc = np.empty((M*N,60))
for i in range(60):
      genre_3_train_mfcc[:,i] = mfcc(genre_3_train[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_3_test_mfcc = np.empty((M*N,37))
for i in range(37):
      genre_3_test_mfcc[:,i] = mfcc(genre_3_test[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_4_train_mfcc = np.empty((M*N,60))
for i in range(60):
      genre_4_train_mfcc[:,i] = mfcc(genre_4_train[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_4_test_mfcc = np.empty((M*N,37))
for i in range(37):
      genre_4_test_mfcc[:,i] = mfcc(genre_4_test[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_5_train_mfcc = np.empty((M*N,60))
for i in range(60):
      genre_5_train_mfcc[:,i] = mfcc(genre_5_train[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_5_test_mfcc = np.empty((M*N,37))
for i in range(37):
      genre_5_test_mfcc[:,i] = mfcc(genre_5_test[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_6_train_mfcc = np.empty((M*N,60))
for i in range(60):
      genre_6_train_mfcc[:,i] = mfcc(genre_6_train[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_6_test_mfcc = np.empty((M*N,37))
for i in range(37):
      genre_6_test_mfcc[:,i] = mfcc(genre_6_test[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_7_train_mfcc = np.empty((M*N,60))
for i in range(60):
      genre_7_train_mfcc[:,i] = mfcc(genre_7_train[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)

genre_7_test_mfcc = np.empty((M*N,37))
for i in range(37):
      genre_7_test_mfcc[:,i] = mfcc(genre_7_test[:,i], samplerate=8000, winstep=0.01, winlen=0.025, numcep=16).reshape(M*N)


U1,s1,V1 = np.linalg.svd(genre_1_train_mfcc,full_matrices=False)
U2,s2,V2 = np.linalg.svd(genre_2_train_mfcc,full_matrices=False)
U3,s3,V3 = np.linalg.svd(genre_3_train_mfcc,full_matrices=False)
U4,s4,V4 = np.linalg.svd(genre_4_train_mfcc,full_matrices=False)
U5,s5,V5 = np.linalg.svd(genre_5_train_mfcc,full_matrices=False)
U6,s6,V6 = np.linalg.svd(genre_6_train_mfcc,full_matrices=False)
U7,s7,V7 = np.linalg.svd(genre_7_train_mfcc,full_matrices=False)

s_x = np.arange(60)
s1_plot = plt.figure()
plt.plot(s_x,s1,'r*')
plt.title("s1 plot")
plt.savefig("subs_learn_mfcc_result/s1_plot.png",dpi=300)

s2_plot = plt.figure()
plt.plot(s_x,s2,'r*')
plt.title("s2 plot")
plt.savefig("subs_learn_mfcc_result/s2_plot.png",dpi=300)

s3_plot = plt.figure()
plt.plot(s_x,s3,'r*')
plt.title("s3 plot")
plt.savefig("subs_learn_mfcc_result/s3_plot.png",dpi=300)

s4_plot = plt.figure()
plt.plot(s_x,s4,'r*')
plt.title("s4 plot")
plt.savefig("subs_learn_mfcc_result/s4_plot.png",dpi=300)

s5_plot = plt.figure()
plt.plot(s_x,s5,'r*')
plt.title("s5 plot")
plt.savefig("subs_learn_mfcc_result/s5_plot.png",dpi=300)

s6_plot = plt.figure()
plt.plot(s_x,s6,'r*')
plt.title("s6 plot")
plt.savefig("subs_learn_mfcc_result/s6_plot.png",dpi=300)

s7_plot = plt.figure()
plt.plot(s_x,s7,'r*')
plt.title("s7 plot")
plt.savefig("subs_learn_mfcc_result/s7_plot.png",dpi=300)

k1 = 55
k2 = 34
k3 = 46
k4 = 50
k5 = 59
k6 = 50
k7 = 58

Uk1 = U1[:,0:k1]
Uk2 = U2[:,0:k2]
Uk3 = U3[:,0:k3]
Uk4 = U4[:,0:k4]
Uk5 = U5[:,0:k5]
Uk6 = U6[:,0:k6]
Uk7 = U7[:,0:k7]


def predict(x):
      orth_proj_norm = np.empty(7)
      for i,Uk in enumerate([Uk1,Uk2,Uk3,Uk4,Uk5,Uk6,Uk7]):
            orth_proj = x - Uk @ (Uk.T @ x)
            orth_proj_norm[i] = np.linalg.norm(orth_proj)
      
      prediction = np.argmin(orth_proj_norm) + 1
      return prediction

misclassified = 0
for i in range(37):
      if predict(genre_1_test_mfcc[:,i]) != 1:
            misclassified += 1
      if predict(genre_2_test_mfcc[:,i]) != 2:
            misclassified += 1
      if predict(genre_3_test_mfcc[:,i]) != 3:
            misclassified += 1
      if predict(genre_4_test_mfcc[:,i]) != 4:
            misclassified += 1
      if predict(genre_5_test_mfcc[:,i]) != 5:
            misclassified += 1
      if predict(genre_6_test_mfcc[:,i]) != 6:
            misclassified += 1
      if predict(genre_7_test_mfcc[:,i]) != 7:
            misclassified += 1
      

accuracy = (37*7 - misclassified)/(37*7)

print(accuracy)






      
