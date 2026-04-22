# from predict_image_utils import build_visualization

# _, _, big_defects, small_defects = build_visualization()

big_defects = ["1"]

if len(big_defects) > 0:
    print("Found Big Defects!")
    print("BIG YOLO Detector")
else:
    print("Start Small YOLO Detector")
