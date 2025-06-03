import cv2
import numpy as np
import os

def separate_objects_contours(image_path, output_dir):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # if big enough
        if w > 50 and h > 50:
            object_img = img[y:y+h, x:x+w]
            
            output_path = os.path.join(output_dir, f"object_{i+1}.png")
            cv2.imwrite(output_path, object_img)
            print(f"Saved: {output_path}")



if __name__ == "__main__":
    image_folder = 'app/props/images'
    sweep_image = 'deliverables/final_report/figures/dalprop_series.jpg'
    #separate_objects_contours(sweep_image, image_folder)

