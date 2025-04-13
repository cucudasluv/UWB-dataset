import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 불러오기
image_path = '/home/adip/road_vertical_line.jpg'
image = cv2.imread(image_path)

# 밝기 조정 함수 추가
def adjust_brightness(image, brightness=0):
    if brightness != 0:
        beta = brightness
        return cv2.convertScaleAbs(image, alpha=1, beta=beta)
    return image

# 흰색 선 검출 및 대표 선만 남기기
def process_image(image, brightness=0, contrast=1.5):
    # 밝기와 대비 조정
    image_adjusted = adjust_brightness(image, brightness)
    image_processed = cv2.convertScaleAbs(image_adjusted, alpha=contrast, beta=20)
    image_processed = cv2.GaussianBlur(image_processed, (5, 5), 0)

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_processed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image_processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # HSV 변환 및 흰색 검출
    hsv = cv2.cvtColor(image_processed, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 모폴로지 연산을 통한 노이즈 제거 및 흰색 선 강조
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def keep_representative_line(mask, image):
    # 컨투어를 찾습니다.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros_like(image)
    
    # 가장 큰 컨투어를 찾습니다.
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 결과 이미지 생성
    result = np.zeros_like(image)
    cv2.drawContours(result, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    
    return result

# 원본 이미지 처리 (어두운 이미지의 밝기 및 대비 조정 추가)
mask = process_image(image, brightness=50, contrast=2.0)

# 이미지에서 대표 선만 남기기
representative_line = keep_representative_line(mask, image)

# 결과 이미지 저장
cv2.imwrite('/home/adip/road_vertical_line_representative.jpg', representative_line)
cv2.imwrite('/home/adip/road_vertical_line_mask.jpg', mask)

# 결과 이미지 표시
plt.figure(figsize=(10, 5))

# 결과 이미지
plt.subplot(1, 2, 1)
plt.title('Representative Line')
plt.imshow(cv2.cvtColor(representative_line, cv2.COLOR_BGR2RGB))

# 마스크 이미지
plt.subplot(1, 2, 2)
plt.title('Mask')
plt.imshow(mask, cmap='gray')

plt.show()

