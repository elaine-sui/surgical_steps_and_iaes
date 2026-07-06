import torchvision.transforms as transforms
import random
from torchvision import transforms

class CustomRandAugment:
    def __init__(self, aug_space):
        self.aug_space = aug_space 
    
    def __call__(self, img):
        return img


class ColorAdjust:
    def __init__(self, min_mag=0.6, max_mag=1.4):
        self.min_mag = min_mag
        self.max_mag = max_mag
    
    def __call__(self, img):
        factor = random.uniform(self.min_mag, self.max_mag)
        return transforms.functional.adjust_saturation(img, factor)
    

class CustomTorchRandAugment:
    def __init__(self, aug_space):
        self.aug_space = aug_space
        self.ShearX_f = transforms.RandomAffine(degrees=0, shear=(-10, 10,0,0))
        self.ShearY_f = transforms.RandomAffine(degrees=0, shear=(0,0,-10, 10))
        self.Rotate_f = transforms.RandomRotation(degrees=20)
        self.TranslateX_f = transforms.RandomAffine(degrees=0, translate=(0.1, 0))
        self.TranslateY_f = transforms.RandomAffine(degrees=0, translate=(0, 0.1))
        self.Equalize_f = transforms.RandomEqualize()
        self.Contrast_f = transforms.RandomAdjustSharpness(sharpness_factor=2)
        self.Color_f = transforms.ColorJitter(saturation=0.3)
        self.Brightness_f = transforms.ColorJitter(brightness=0.2)
        self.Sharpness_f = transforms.RandomAdjustSharpness(sharpness_factor=2)
        self.k = 3
    
    def shear_x(self, img):
        return self.ShearX_f(img)
    
    def shear_y(self, img):
        return self.ShearY_f(img)
    
    def rotate(self, img):
        return self.Rotate_f(img)
    
    def translate_x(self, img):
        return self.TranslateX_f(img)
    
    def translate_y(self, img):
        return self.TranslateY_f(img)
    
    def autocontrast(self, img):
        return transforms.functional.autocontrast(img)
    
    def equalize(self, img):
        return self.Equalize_f(img)
    
    def adjust_contrast(self, img):
        return transforms.functional.adjust_contrast(img, 2)
    
    def adjust_color(self, img):
        return self.Color_f(img)
    
    def adjust_brightness(self, img):
        return self.Brightness_f(img)
    
    def adjust_sharpness(self, img):
        return self.Sharpness_f(img)

    def __call__(self, img):
        selected_element = random.choices(list(range(0,self.k+1)), [1/10,2/10,3/10,4/3])[0]
        if selected_element==0:
            return img
        aug_op = random.sample(self.aug_space, k=selected_element)  # 随机选择增强操作
        
        for op in aug_op:
            aug_type = op[0]['type']
            
            if aug_type == 'ShearX':
                img = self.shear_x(img)
            elif aug_type == 'ShearY':
                img = self.shear_y(img)
            elif aug_type == 'Rotate':
                img = self.rotate(img)
            elif aug_type == 'TranslateX':
                img = self.translate_x(img)
            elif aug_type == 'TranslateY':
                img = self.translate_y(img)
            elif aug_type == 'AutoContrast':
                img = self.autocontrast(img)
            elif aug_type == 'Equalize':
                img = self.equalize(img)
            elif aug_type == 'Contrast':
                img = self.adjust_contrast(img)
            elif aug_type == 'Color':
                img = self.adjust_color(img)
            elif aug_type == 'Brightness':
                img = self.adjust_brightness(img)
            elif aug_type == 'Sharpness':
                img = self.adjust_sharpness(img)
        
        return img