import numpy as np
import cv2
from tensorflow import keras
import sys

sys.stdout.reconfigure(encoding='utf-8')


threshold = 0.75  # THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
model = keras.models.load_model('traffic_sign_model.h5')


def preprocess_img(imgBGR, erode_dilate=True):  # tiền xử lý để phát hiện các dấu hiệu trong hình ảnh.
    rows, cols, _ = imgBGR.shape
    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    Bmin = np.array([100, 43, 46])
    Bmax = np.array([124, 255, 255])
    img_Bbin = cv2.inRange(imgHSV, Bmin, Bmax)

    Rmin1 = np.array([0, 43, 46])
    Rmax1 = np.array([10, 255, 255])
    img_Rbin1 = cv2.inRange(imgHSV, Rmin1, Rmax1)

    Rmin2 = np.array([156, 43, 46])
    Rmax2 = np.array([180, 255, 255])
    img_Rbin2 = cv2.inRange(imgHSV, Rmin2, Rmax2)
    img_Rbin = np.maximum(img_Rbin1, img_Rbin2)
    img_bin = np.maximum(img_Bbin, img_Rbin)

    if erode_dilate is True:
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        img_bin = cv2.erode(img_bin, kernelErosion, iterations=2)
        img_bin = cv2.dilate(img_bin, kernelDilation, iterations=2)

    return img_bin


def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return rects

    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            if 1.0 * w / h < wh_ratio and 1.0 * h / w < wh_ratio:
                rects.append([x, y, w, h])
    return rects


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Gioi han toc do: 20 km/h'
    elif classNo == 1:
        return 'Gioi han toc do: 30 km/h'
    elif classNo == 2:
        return 'Gioi han toc do: 50 km/h'
    elif classNo == 3:
        return 'Gioi han toc do: 60 km/h'
    elif classNo == 4:
        return 'Gioi han toc do: 70 km/h'
    elif classNo == 5:
        return 'Gioi han toc do: 80 km/h'
    elif classNo == 6:
        return 'Ket thuc gioi han toc do: 80 km/h'
    elif classNo == 7:
        return 'Gioi han toc do: 100 km/h'
    elif classNo == 8:
        return 'Gioi han toc do: 120 km/h'
    elif classNo == 9:
        return 'Cam vuot'
    elif classNo == 10:
        return 'Phuong tien tren 3.5 tan khong duoc di qua'
    elif classNo == 11:
        return 'Quyen uu tien tai giao lo tiep theo'
    elif classNo == 12:
        return 'Duong uu tien'
    elif classNo == 13:
        return 'Nhuong duong'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'Cam phuong tien'
    elif classNo == 16:
        return 'Cam xe tren 3.5 tan'
    elif classNo == 17:
        return 'Cam vao'
    elif classNo == 18:
        return 'Than trong chung'
    elif classNo == 19:
        return 'Khuc cua nguy hiem ben trai'
    elif classNo == 20:
        return 'Khuc cua nguy hiem ben phai'
    elif classNo == 21:
        return 'Duong cong doi'
    elif classNo == 22:
        return 'Duong gap genh'
    elif classNo == 23:
        return 'Duong tron'
    elif classNo == 24:
        return 'Duong hep'
    elif classNo == 25:
        return 'Lam duong'
    elif classNo == 26:
        return 'Tin hieu giao thong'
    elif classNo == 27:
        return 'Nguoi di bo'
    elif classNo == 28:
        return 'Tre em di duong'
    elif classNo == 29:
        return 'Xe dap qua duong'
    elif classNo == 30:
        return 'Coi chung bang/tuyet'
    elif classNo == 31:
        return 'Dong vat hoang da'
    elif classNo == 32:
        return 'Ket thuc tat ca gioi han toc do va vuot qua'
    elif classNo == 33:
        return 'Re phai ve phia truoc'
    elif classNo == 34:
        return 'Re trai ve phia truoc'
    elif classNo == 35:
        return 'Di thang'
    elif classNo == 36:
        return 'Di thang hoac phai'
    elif classNo == 37:
        return 'Di thang hoac trai'
    elif classNo == 38:
        return 'Di ben phai'
    elif classNo == 39:
        return 'Di ben trai'
    elif classNo == 40:
        return 'Bat buoc di vong xuyen'
    elif classNo == 41:
        return 'Ket thuc - Khong di qua'
    elif classNo == 42:
        return 'Het duong cam xe 3.5 tan'


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while (1):
        ret, img = cap.read()
        img_bin = preprocess_img(img, False)
        cv2.imshow("bin image", img_bin)
        min_area = img_bin.shape[0] * img.shape[1] / (25 * 25)
        rects = contour_detect(img_bin, min_area=min_area)   # lấy x, y, h và w.
        img_bbx = img.copy()
        for rect in rects:
            xc = int(rect[0] + rect[2] / 2)
            yc = int(rect[1] + rect[3] / 2)

            size = max(rect[2], rect[3])
            x1 = max(0, int(xc - size / 2))
            y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2))
            y2 = min(rows, int(yc + size / 2))

            # rect[2] là chiều rộng và rect[3] là chiều cao
            if rect[2] > 100 and rect[3] > 100:             #chỉ phát hiện những biển báo có chiều cao và chiều rộng >100
                cv2.rectangle(img_bbx, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
            crop_img = np.asarray(img[y1:y2, x1:x2])
            crop_img = cv2.resize(crop_img, (32, 32))
            crop_img = preprocessing(crop_img)
            cv2.imshow("afterprocessing", crop_img)
            crop_img = crop_img.reshape(1, 32, 32, 1)       # (1,32,32) sau khi định hình lại nó trở thành (1,32,32,1)
            predictions = model.predict(crop_img)           # Đưa ra dự đoán
            classIndex = np.argmax(predictions,axis=1)
            probabilityValue = np.amax(predictions)
            if probabilityValue > threshold:
                #viết tên lớp trên màn hình đầu ra
                cv2.putText(img_bbx, str(classIndex) + " " + str(getCalssName(classIndex)), (rect[0], rect[1] - 10),
                            font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                # ghi giá trị xác suất trên màn hình đầu ra
                cv2.putText(img_bbx, str(round(probabilityValue * 100, 2)) + "%", (rect[0], rect[1] - 40), font, 0.75,
                            (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("detect result", img_bbx)
        if cv2.waitKey(1) & 0xFF == ord('q'):           # q để thoát tất cả
            break
cap.release()
cv2.destroyAllWindows()