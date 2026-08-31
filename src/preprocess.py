import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    """
    Remove uninformative black borders around retinal circle.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img  # Return original if crop fails
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def ben_graham_preprocessing(image_path, img_size=224, sigma_x=10):
    """
    Applies circle crop and Gaussian blur weighted subtraction for contrast.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Crop uninformative dark borders
    image = crop_image_from_gray(image)
    
    # 2. Resize to standard square input
    image = cv2.resize(image, (img_size, img_size))
    
    # 3. Add Gaussian blur subtraction for enhanced vessel clarity
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma_x), -4, 128)
    
    return image