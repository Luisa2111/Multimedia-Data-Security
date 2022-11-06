def detection_caller(name_originalImage, name_watermarkedImage, name_attackedImage):
    if name_watermarkedImage[24:-9] == "pixel":
        import detection_pixel as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "ef26420c":
        import detection_ef26420c as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "youshallnotmark":
        import detection_youshallnotmark as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "blitz":
        import detection_blitz as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "omega":
        import detection_howimetyourmark as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "howimetyourmark":
        import detection_omega as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "weusedlsb":
        import detection_weusedlsb as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "thebavarians":
        import detection_thebavarians as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "theyarethesamepicture":
        import detection_theyarethesamepicture as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[24:-9] == "dinkleberg":
        import detection_dinkleberg as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)
    elif name_watermarkedImage[:-9] == "failedfouriertransform":
        import detection_failedfouriertransform as det
        return det.detection(name_originalImage, name_watermarkedImage, name_attackedImage)

    print("something went wrong in detection_caller")
    return 1
