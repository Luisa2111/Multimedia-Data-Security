import numpy as np
from psnr import similarity, wpsnr
import cv2
import matplotlib.pyplot as plt
import image_processing as ip
import detection_howimetyourmark as det

name_groups_competitors=["ef26420c", "pixel", "youshallnotmark",
						 "blitz", "omega", "howimetyourmark", "weusedlsb", "thebavarians",
						 "theyarethesamepicture", "dinkleberg", "failedfouriertransform"]
name_images_competitors=["buildings", "rollercoaster", "tree"]
if __name__ == "__main__":
    name_image = "./images_of_competition/howimetyourmark_buildings.bmp"
    name_original = "./sample_images_roc/buildings.bmp"
    #name_attacked = name_original + "attacked.bmp"

    original = cv2.imread(name_original, 0)
    image = cv2.imread(name_image, 0)
    """std = 0.21000000000000005
    seed = 123
    attacked = ip.resiz(image, std, seed)
    cv2.imwrite(name_attacked, attacked)"""

    name_attacked = "./attacked_finished/howimetyourmar/ef26420c_howimetyourmark_buildings.bmp"

    """wpsnr = ip.wpsnr(attacked, image)
    print(wpsnr)"""

    dete,w = det.detection(name_original, name_image, name_attacked)
    print(dete)

