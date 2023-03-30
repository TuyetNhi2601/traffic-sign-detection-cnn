# Traffic sign detection CNN.
## Dataset nằm trong đường dẫn này: https://github.com/Ali-Jakhar/Traffic-Sign-Detection-And-Recognition
* Bộ dataset German Traffic Sign Recognition Benchmark
Do không có nhiều thời gian nên việc thu thập dữ liệu là rất khó, vì vậy tôi đã sử dụng bộ dữ liệu nhận diện biển báo giao thông có sẵn “German Traffic Sign Recognition Benchmark”.
- Tập data bao gồm hơn 39,209 ảnh được chụp trên đường phố nước Đức.
- Chia thành 43 lớp loại khác nhau của Traffic-sign.
- Kích thước của traffic signs trong tập này dao động từ 15×15 đến 222×193 pixel.
* Phân phối tập dữ liệu biển báo giao thông.

![image](https://user-images.githubusercontent.com/122681319/228909630-26408b86-5034-431f-8c1a-56438184f70e.png)

Một số loại biển báo trong tập dữ liệu như: 

![image](https://user-images.githubusercontent.com/122681319/228909730-a3868094-8896-4795-9597-392c81cde5aa.png)

## Huấn luyện mô hình

Mô hình CNN được áp dụng trong chương trình này:

![image](https://user-images.githubusercontent.com/122681319/228909989-14b59111-c63c-4058-bf75-1ec9a77e43d5.png)

Đối với mô hình đào tạo, tôi đã đặt một số tham số như batch size, bước trên mỗi epoch và số lượng epoch với: batch_size_val=30, steps_per_epoch_val=500, epochs_val=40.

## Kết quả 
* Phát hiện và phân loại được biển báo trong thời gian thực.

![image](https://user-images.githubusercontent.com/122681319/228910293-b3ab0fd5-f5f6-4bce-8105-39a23c47b1b1.png)

## Đánh giá mô hình
* Với mô hình CNN được đào tạo, chương trình đã đạt được kết quả đáng mong đợi, đồ thị đào tạo và tổn thất thể hiện trong hình bên dưới.

![image](https://user-images.githubusercontent.com/122681319/228910441-5492b2d5-33fe-49ba-9336-4f4ce8ed8efb.png)

* Để đánh giá mô hình, tôi đã chuyển 20% dữ liệu thử nghiệm cho mô hình để thử nghiệm. Tôi đã đạt được 99% test accuracy.

![image](https://user-images.githubusercontent.com/122681319/228910578-570c505a-6f5d-4e11-b38e-e8e6358a9d2f.png)

Tham khảo: https://www.youtube.com/watch?v=BvYFGScsE7Y
