import SimpleITK as sitk
from delira.utils.imageops import sitk_image_to_data
import matplotlib.pyplot as plt

def read_meta_from_dcm(img_file):
    # Print out meta information (keys + values)
    reader = sitk.ImageFileReader()
    reader.SetFileName(img_file)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        #if k == "0020|000d" or k == "0020|000e":
        print("({0}) = = \"{1}\"".format(k,v))

    print("Image Size: {0}".format(reader.GetSize()))
    print("Image PixelType: {0}".format(sitk.GetPixelIDValueAsString(reader.GetPixelID())))

def plot_image_from_dcm(img_file):
    img_sitk = sitk.ReadImage(img_file)
    img = sitk_image_to_data(img_sitk)

    imgplot = plt.imshow(img[:, :, 0], cmap=plt.get_cmap("Greys"))
    plt.show()


#img_file = "/home/students/moriz/PycharmProjects/MA_my_code/test/CBIS-ddsm/Calc-Test_P_00038_LEFT_CC/08-29-2017-ddsm-96009/1-full mammogram images-63992/000000.dcm"
img_file = "/home/students/moriz/PycharmProjects/MA_my_code/test/CBIS-ddsm/Calc-Test_P_00038_LEFT_CC_1/08-29-2017-ddsm-94942/1-ROI mask images-18515/000000.dcm"

read_meta_from_dcm(img_file)
plot_image_from_dcm(img_file)