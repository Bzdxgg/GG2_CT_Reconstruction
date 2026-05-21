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
    """
    Test 1: non-ideal-source reconstruction visualisation and profile comparison.
    
    We use a non-ideal source to reconstruct a type-6 hip phantom
    with and without filtering, and compare the reconstructed central-row
    profiles of circle and point phantoms.
    This test is intended for qualitative inspection of reconstruction behaviour
    under realistic imaging conditions. 

    RESULTS ANALYSIS:
    1. Filtered vs unfiltered reconstruction:
    The filtered reconstruction recovers sharper edges and structural
    detail, while the unfiltered reconstruction appears blurred.

    2. Circle vs point phantom profiles:
    The circle phantom produces a broader intensity distribution across the
    central row, whereas the point phantom produces a more
    localised peak concentrated over fewer pixels.
    """
    
    mat = Material()
    src = Source()
    s = src.photon('100kVp, 2mm Al')
    
    # Visualise Type 6 Phantom with and without filter
    p_6 = ct_phantom(mat.name, 256, 6)
    y_6_filtered = scan_and_reconstruct(s, mat, p_6, 0.01, 256, use_filter=True)
    y_6_unfiltered = scan_and_reconstruct(s, mat, p_6, 0.01, 256, use_filter=False)
    save_draw(y_6_filtered, 'results', 'test_1_filtered', title='Filtered Type 6 (Default Materials), 100kVp, 2mm Al')
    save_draw(y_6_unfiltered, 'results', 'test_1_unfiltered', title='Unfiltered Type 6 (Default Materials), 100kVp, 2mm Al')
    save_draw(p_6, 'results', 'test_1_phantom', title='Type 6 (Default Materials) Phantom')

    # Compare profile response of extended and localised structures
    p_1 = ct_phantom(mat.name, 256, 1)
    p_2 = ct_phantom(mat.name, 256, 2)
    s = src.photon('80kVp, 1mm Al')
    y_1_filtered = scan_and_reconstruct(s, mat, p_1, 0.01, 256)
    y_2_filtered = scan_and_reconstruct(s, mat, p_2, 0.01, 256)

    # save some meaningful results
    save_plot(y_1_filtered[128,:], 'results', 'test_2_plot_Circle', title='Central Row Profile of Filtered Circle Phantom (Default Materials), 80kVp, 1mm Al')
    save_plot(y_2_filtered[128,:], 'results', 'test_2_plot_Point', title='Central Row Profile of Filtered Point Phantom (Default Materials), 80kVp, 1mm Al')



def test_2():
    """ Test 2: ideal-source bone phantom mean-value check.
    We use an ideal source to scan a bone phantom and print the 
    theoretical vs measured mean attenuation in the central ROI.
    This is for numeric sanity checks rather than visual inspection.
    Ideal coefficient and measured mean value are outputed as text file.

    RESULTS ANALYSIS:
    1. Bone (~0.25% error): Perfect match confirms that the reconstruction is fully calibrated.
    2. Titanium (~63.3% underestimation): Even though the original phantom is a perfectly uniform circle,
       the very high metal density causes strong photon starvation through the center. This leads to a
       pronounced cupping artifact, which significantly lowers the reconstructed mean values in the central ROI.
    """

    mat = Material()
    src = Source()
    p_titanium = ct_phantom(mat.name, 256, 1, "Titanium")
    p_bone = ct_phantom(mat.name, 256, 1, "Bone")
    s = fake_source(src.mev, 0.1, method='ideal')
    y_titanium = scan_and_reconstruct(s, mat, p_titanium, 0.1, 256)
    y_bone = scan_and_reconstruct(s, mat, p_bone, 0.1, 256)

    # save some meaningful results
    mean_val_titanium = np.mean(y_titanium[64:192, 64:192])
    mean_val_bone = np.mean(y_bone[64:192, 64:192])
    idx = mat.mev.tolist().index(0.07)
    # print(f"Ideal value is {mat.coeff('Bone')[idx]}")
    # print(f"Mean value is {mean_val}")
    
    save_draw(y_titanium, 'results', 'test_2_reconstruction_titanium')
    save_draw(y_bone, 'results', 'test_2_reconstruction_bone')


    with open('results/test_2_summary.txt', 'w') as f:
        f.write('Test 2: calibration circle phantom\n')
        f.write('Ideal value for Titanium: ' + str(mat.coeff('Titanium')[idx]) + '\n')
        f.write('Mean value for Titanium: ' + str(mean_val_titanium) + '\n')
        f.write('Ideal value for Bone: ' + str(mat.coeff('Bone')[idx]) + '\n')
        f.write('Mean value for Bone: ' + str(mean_val_bone) + '\n')

def test_3():
    """ Test 3: Investigate the Effect of Angles
    Point Filtered Reconstruction with different angles (ideal source, with threshold)

    RESULTS ANALYSIS:
    Angles < 256: Using fewer than 256 projection angles causes strong angular aliasing 
    because there aren't enough views to fully sample the image. This leaves gaps in the 
    Radon transform data, so the back-projected rays don't cancel properly outside the center. 
    The result is the noticeable radial “starburst” streaking artifact.
    Angles ≥ 256: At 256 or more projection angles, the sampling becomes dense enough for 
    a 256x256 grid. The back-projected rays combine much more cleanly, background artifacts 
    are greatly reduced, and the central point is reconstructed correctly.
    """
    mat = Material()
    src = Source()
    p = ct_phantom(mat.name, 256, 2)
    s = fake_source(src.mev, 0.1, method='ideal')
    save_draw(p, 'results', 'test_3_phantom')
    
    angles = [32, 64, 128, 256, 512]
    for angle in angles:
        y = scan_and_reconstruct(s, mat, p, 0.1, angle)
        save_draw(y, 'results', f'test_3_reconstruction_{angle}', caxis=[0,0.05*np.max(y)])
        
def test_4(): 
    """ Test 4: Investigate the Effect of Different Interpolation Methods and Orders
    """
    mat = Material()
    src = Source()
    p = ct_phantom(mat.name, 256, 2)
    save_draw(p, 'results', 'test_4_phantom')
    
    s = fake_source(src.mev, 0.1, method='ideal')
    
    orders = [0, 1, 3]
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for order in orders:
        y = scan_and_reconstruct(
            s, mat, p, 0.1, 256,
            use_filter=True,
            order=order,
            skip=1,
            use_interp1d=False
        )

        ax.plot(y[128, :], label=f'Order {order}')

    # Plot linear interpolation result
    y_lin = scan_and_reconstruct(
        s, mat, p, 0.1, 256,
        use_filter=True,
        order=1,
        skip=1,
        use_interp1d=True
    )

    ax.set_xlim(120, 135)   # adjust as needed
    ax.set_title('Reconstruction Comparison')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/test_4_combined.png')
    plt.close()
        
    for order in orders:
        y = scan_and_reconstruct(s, mat, p, 0.1, 256, use_filter=True, order=order, skip=1, use_interp1d=False)
        # save_draw(y, 'results', f'test_4_reconstruction_order_{order}', caxis=[0,0.05*np.max(y)])
        save_plot(y[128,:], 'results', f'test_4_reconstruction_order_{order}')
    
    y_lin = scan_and_reconstruct(s, mat, p, 0.1, 256, use_filter=True, order=1, skip=1, use_interp1d=True)
    # save_draw(y_lin, 'results', 'test_4_reconstruction_linear', caxis=[0,0.05*np.max(y_lin)])
    save_plot(y_lin[128,:], 'results', 'test_4_reconstruction_linear')

def test_5():
    """ Test 5: Investigate the Effect of Alpha in Ramp Filter
    """
    mat = Material()
    src = Source()
    p = ct_phantom(mat.name, 256, 2)
    s = fake_source(src.mev, 0.1, method='ideal')
    save_draw(p, 'results', 'test_5_phantom')
    
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
    for alpha in alphas:
        y = scan_and_reconstruct(s, mat, p, 0.1, 256, use_filter=True, alpha=alpha)
        # save_draw(y, 'results', f'test_5_reconstruction_alpha_{alpha}', caxis=[0,0.05*np.max(y)])
        alpha_str = str(alpha).replace('.', '_')
        save_plot(y[128,:], 'results', f'test_5_reconstruction_alpha_{alpha_str}')
        
        
# print('Test 1')
# test_1()
# print('Test 2')
# test_2()
print('Test 3')
test_3()
# print('Test 4')
# test_4()
# print('Test 5')
# test_5()
