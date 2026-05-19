import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *
from ct_calibrate import *

def test_detect():
    mat = Material()
    src = Source()
    a = ct_detect(src.photon('100kVp, 2mm Al'), mat.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
    plot(np.log(a))

    fake_a = fake_source(mat.mev, 0.12, mat.coeff('Aluminium'), 2, 'ideal')
    a = ct_detect(fake_a, mat.coeff('Water'), np.arange(0, 10.1, 0.1), 1)
    plot(np.log(a))

def test_sinogram(i=1):
    mat = Material()
    src = Source()
    M = ct_phantom(mat.name, 256, i, 'Bone')
    draw(M)
    p = ct_scan(src.photon('100kVp, 2mm Al'), mat, M, 0.1, 256)

    p_calibrated = ct_calibrate(src.photon('100kVp, 2mm Al'), mat, p, 0.02)

    draw(p_calibrated)
    plot(p_calibrated[0, :])

# def test_calibrate(i=2):
#     mat = Material()
#     src = Source()
#     M = ct_phantom(mat.name, 256, i, 'Bone')
#     p = ct_scan(src.photon('100kVp, 2mm Al'), mat, M, 0.1, 256)
#     p_calibrated = ct_calibrate(src.photon('100kVp, 2mm Al'), mat, p, 0.02)
#     peak = np.max(p_calibrated)
    
#     print('Measured peak:', peak)

def test_calibrate(i=2):

    mat = Material()
    src = Source()

    scale = 0.1 
    photons = src.photon('100kVp, 2mm Al')
    coeff = mat.coeff('Bone')

    M_bone = ct_phantom(mat.name, 256, i, 'Bone')
    p_bone = ct_scan(photons, mat, M_bone, scale, 256)
    p_calibrated_bone = ct_calibrate(photons, mat, p_bone, scale)
    
    measured_bone_attenuation = np.max(p_calibrated_bone)
    print(f"Max Attenuation for Bone: {measured_bone_attenuation:.6f}")

    I0_ideal = np.sum(photons)
    I_ideal = np.sum(photons * np.exp(-coeff * scale))
    p_ideal = -np.log(I_ideal / I0_ideal)
    print(f"Ideal theoretical attenuation for Bone: {p_ideal:.6f}")
    
    return p_calibrated_bone

def test_calibrate_scales(i=2):
    mat = Material()
    src = Source()

    photons = src.photon('100kVp, 2mm Al')
    coeff = mat.coeff('Bone')

    # Define a range of scales (thicknesses in cm) to test
    scales = np.linspace(0.01, 0.5, 10) 
    
    measured_attenuations = []
    ideal_attenuations = []

    # The phantom matrix doesn't change with scale, so generate it once to save time
    M_bone = ct_phantom(mat.name, 256, i, 'Bone')

    print("Running scans over multiple scales...")
    for scale in scales:
        print(f"Testing scale: {scale:.3f} cm...")
        
        # Perform the scan and calibration for the current scale
        p_bone = ct_scan(photons, mat, M_bone, scale, 256)
        p_calibrated_bone = ct_calibrate(photons, mat, p_bone, scale)
        
        # 1. Get measured max attenuation
        measured_bone_attenuation = np.max(p_calibrated_bone)
        measured_attenuations.append(measured_bone_attenuation)

        # 2. Calculate ideal theoretical attenuation
        I0_ideal = np.sum(photons)
        I_ideal = np.sum(photons * np.exp(-coeff * scale))
        p_ideal = -np.log(I_ideal / I0_ideal)
        ideal_attenuations.append(p_ideal)

    # Plot the results
    plt.figure(figsize=(8, 6))
    
    # Plot measured values (blue circles with solid line)
    plt.plot(scales, measured_attenuations, 'bo-', label='Measured (ct_calibrate)', markersize=6)
    
    # Plot ideal values (red crosses with dashed line)
    plt.plot(scales, ideal_attenuations, 'rx--', label='Ideal (Theoretical)', markersize=8)
    
    plt.title('Total Attenuation vs. Scale (Thickness) for Bone')
    plt.xlabel('Scale (cm per pixel)')
    plt.ylabel('Total Attenuation (p)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return scales, measured_attenuations, ideal_attenuations

scales, measured, ideal = test_calibrate_scales()
# test_calibrate()
# test_sinogram(2)
