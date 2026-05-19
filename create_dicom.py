import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
import os


def create_dicom(x, filename, sp, sz=None, f=1, study_uid=None, series_uid=None, frame_uid=None, time=datetime.datetime.now(), storage_directory=None):

	""" Create DICOM format output file from data

	create_dicom(x, filename, sp) creates a new DICOM file with a
	name `filename_0001.dcm' and containing data from x. The pixel scale is
	given by sp which is in mm.

	create_dicom(x, filename, sp, sz, f) creates a new DICOM file with a
	name formed from the given filename and the frame number f, and
	containing data from x. The pixel scale is given by sp, and the frame
	spacing is given by sz, both of which are in mm.

	create_dicom(x, filename, sp, sz, f, study_uid, series_uid, time)
	uses the DICOM UIDs study_uid and series_uid, and also the
	datetime, for the file. This is useful if you want to write several
	frames in the same DICOM series. The UIDs can be generated
	using the DICOMUID function. The time can be generated using datetime.datetime.now().

	optional storage_directory parameter can set the file's storage directory path
	"""

	# check for inputs
	if sz is None:
		sz = sp

	if study_uid is None:
		study_uid = pydicom.uid.generate_uid()

	if series_uid is None:
		series_uid = pydicom.uid.generate_uid()

	if frame_uid is None:
		frame_uid = pydicom.uid.generate_uid()


	# get data with the appropriate limits
	x = x + 1024
	x = np.clip(x, 0, None)
	x = np.clip(x, None, 4096)

	# Initial write to create DICOM file with default settings
	full_filename = filename + '_' + str(f).zfill(4) + '.dcm'
	full_file = full_filename

	#add storage directory if needed
	if storage_directory is not None:
		full_filename = os.path.join(storage_directory, full_filename)

	series_date = time.strftime('%Y%m%d')
	series_time = time.strftime('%H%M%S.%f')
	nowtime = datetime.datetime.now()

	# necessary tags
	ds = FileDataset(full_file, {}, file_meta=Dataset(), preamble=b"\0"*128)
	ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
	ds.Modality = 'CT'
	ds.ContentDate = nowtime.strftime('%Y%m%d')
	ds.ContentTime = nowtime.strftime('%H%M%S.%f')
	ds.AcquisitionDate = nowtime.strftime('%Y%m%d')
	ds.AcquisitionTime = nowtime.strftime('%H%M%S.%f')
	ds.StudyInstanceUID =  study_uid
	ds.SeriesInstanceUID = series_uid
	ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
	ds.SOPInstanceUID = pydicom.uid.generate_uid()
	ds.FrameOfReferenceUID = frame_uid
	ds.StudyDescription = 'GG2 Study ' + study_uid[56:]
	ds.SeriesDescription = 'GG2 Series ' + series_uid[56:]
	ds.StudyID = '1'
	ds.SeriesNumber = 1
	ds.StudyDate = series_date
	ds.SeriesDate = series_date
	ds.StudyTime = series_time
	ds.SeriesTime = series_time
	ds.PatientName = 'GG2 Patient'
	ds.RescaleIntercept = '-1024'
	ds.RescaleSlope = '1'
	ds.RescaleType = 'HU'
	ds.WindowWidth = '2000'
	ds.WindowCenter = '0'
	ds.ImagePositionPatient = [0.000, 0.000, round(float(f * sz), 8)]
	ds.ImageOrientationPatient = [1.000, 0.000, 0.000, 0.000, 1.000, 0.000]
	ds.SpacingBetweenSlices = str(round(sz, 8))
	ds.SliceThickness = str(round(sz, 8))
	ds.GantryDetectorTilt = '0'
	ds.SliceLocation = str(round(f * sz, 8))
	ds.PixelSpacing = [sp, sp]

	## These are the necessary imaging components of the FileDataset object.
	ds.SamplesPerPixel = 1
	ds.PhotometricInterpretation = "MONOCHROME2"
	ds.PixelRepresentation = 0
	ds.HighBit = 15
	ds.BitsStored = 16
	ds.BitsAllocated = 16
	ds.Columns = x.shape[1]
	ds.Rows = x.shape[0]

	if x.dtype != np.uint16:
		x = x.astype(np.uint16)

	ds.PixelData = x.tostring()

	# write final file with this metadata
	ds.save_as(full_filename, write_like_original=False)


def read_dicom(filename):

	""" Read DICOM format input file to data

	[x, sp] = read_dicom(filename) reads a DICOM file with a name 
	`filename' info the (int16) data array x. The pixel scale is also
	returned in sp which is in mm."""
	
	ds = pydicom.filereader.dcmread(filename)
	x = (ds.pixel_array + ds.RescaleIntercept).astype(np.int16)
	sp = ds.PixelSpacing[0]
    
	return x, sp

