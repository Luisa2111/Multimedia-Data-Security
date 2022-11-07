

def detection_caller(name_originalImage, name_watermarkedImage, name_attackedImage):
    if "pixel" in name_watermarkedImage[24:]:
        import detection_pixel as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "ef26420c" in name_watermarkedImage[24:]:
        import detection_ef26420c as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "youshallnotmark" in name_watermarkedImage[24:]:
        import detection_youshallnotmark as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "blitz" in name_watermarkedImage[24:]:
        import detection_blitz as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "omega" in name_watermarkedImage[24:]:
        import detection_howimetyourmark as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "howimetyourmark" in name_watermarkedImage[24:]:
        import detection_omega as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "weusedlsb" in name_watermarkedImage[24:]:
        import detection_weusedlsb as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "thebavarians" in name_watermarkedImage[24:]:
        import detection_thebavarians as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "theyarethesamepicture" in name_watermarkedImage[24:]:
        import detection_theyarethesamepicture as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "dinkleberg" in name_watermarkedImage[24:]:
        import detection_dinkleberg as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif "failedfouriertransform" in name_watermarkedImage[:]:
        import detection_failedfouriertransform as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)

    print("something went wrong in detection_caller")
    return 1
