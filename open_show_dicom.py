import pydicom as dicom
import matplotlib.pylab as plt

# specify your image path
image_path = '1442452be09c.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array)