from tracemalloc import _TraceTuple
import cv2 
import numpy as np


def sketch_extractor(image_path:str="race_car.jpeg", save_path:str="sketched_image.jpeg", blur_kernel:_TraceTuple = (5,5), sigma_x:int = 0 ):
    """Generates sketch drawing form original image.

    Parameters
    ----------
    image_path : str, optional
        Source image path wgşhc will be used to transform, by default "race_car.jpeg"
    save_path : str, optional
        Saving path of the output, by default "sketched_image.jpeg"
    """

    try:
        # reading original image
        image = cv2.imread(image_path)

        # converting the image to grayscale 
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # inverting the pixel intensity
        invert = cv2.bitwise_not(grey_img)

        # bluring for removing the shapes
        blur = cv2.GaussianBlur(invert, blur_kernel, sigma_x)

        # inverting the pixel intensity again
        invertedblur = cv2.bitwise_not(blur)


        sketch = cv2.divide(grey_img, invertedblur, scale=256.0)
        cv2.imwrite(save_path, sketch)


        total_img = np.concatenate((grey_img, invert, invert, blur, invertedblur, sketch), axis=1)
        cv2.imwrite("transforming_steps", total_img)

        cv2.imshow('transforming_steps', total_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")