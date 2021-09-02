from scipy import io as sio
import numpy as np

mat = sio.loadmat("polydata_o26_v7.mat")
with open('polydata.npy', 'wb') as f:
    for n in range(2,7):
        m = mat['Poly'+str(n)][0]
        dat = np.zeros(len(m), dtype=[('coef', 'f8', n+1), ('root', 'c16', n)])
        for i in range(len(m)):
            dat[i][0] = m[i][0][::-1,0]
            dat[i][1] = m[i][1][::-1,0]
        np.save(f, dat)




# print(dat[7]['coef'])
# print(dat['coef'][7])
# print(dat[7][0])

# with open('polydata.npy', 'wb') as f:
#     np.save(f, mat['Poly2'])
#     np.save(f, mat['Poly3'])
#     np.save(f, mat['Poly4'])
#     np.save(f, mat['Poly5'])
#     np.save(f, mat['Poly6'])
#
# with open('polydata.npy', 'rb') as f:
#     p2 = np.load(f, allow_pickle=True)
#     p3 = np.load(f, allow_pickle=True)
#     p4 = np.load(f, allow_pickle=True)
#     p5 = np.load(f, allow_pickle=True)
#     p6 = np.load(f, allow_pickle=True)
