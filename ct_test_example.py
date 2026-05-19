import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
from ct_calibrate import *

def test_1():

    mat = Material()
    src = Source()
    p = ct_phantom(mat.name, 256, 6)
    s = src.photon('100kVp, 2mm Al')
    y = scan_and_reconstruct(s, mat, p, 0.01, 256)

    # save some meaningful results
    save_draw(y, 'results', 'test_1_image')
    save_draw(p, 'results', 'test_1_phantom')



def test_2():

    # work out what the initial conditions should be
    mat = Material()
    src = Source()
    p1 = ct_phantom(mat.name, 256, 1)
    p2 = ct_phantom(mat.name, 256, 2)
    s = src.photon('80kVp, 1mm Al')
    y1 = scan_and_reconstruct(s, mat, p1, 0.01, 256)
    y2 = scan_and_reconstruct(s, mat, p2, 0.01, 256)

    # save some meaningful results
    save_plot(y1[128,:], 'results', 'test_2_plot_Circle')
    save_plot(y2[128,:], 'results', 'test_2_plot_Point')




def test_3():
    # work out what the initial conditions should be
    mat = Material()
    src = Source()
    p = ct_phantom(mat.name, 256, 1, 'Bone')
    s = fake_source(src.mev, 0.1, method='ideal')
    y = scan_and_reconstruct(s, mat, p, 0.1, 256)

    # save some meaningful results
    mean_val = np.mean(y[64:192, 64:192])
    idx = mat.mev.tolist().index(0.07)
    print(f"Ideal value is {mat.coeff('Bone')[idx]}")
    print(f"Mean value is {mean_val}")
    
# print('Test 1')
# test_1()
# print('Test 2')
# test_2()
print('Test 3')
test_3()
