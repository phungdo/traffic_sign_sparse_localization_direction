# KITTI-TS_dataset_report

# Báo cáo dataset KITTI-TS

## 1. Tổng quan nhanh

Dataset `KITTI-TS` là một bộ dữ liệu ảnh giao thông theo phong cách KITTI, tập trung vào bài toán nhận diện/phát hiện biển báo giao thông. Dữ liệu chính gồm ảnh `.png` và nhãn `.json`.

Nếu bỏ qua thư mục phụ do macOS tạo ra (`__MACOSX`) và các file `.DS_Store`, dataset chính có:

| Thành phần | Số lượng |
| --- | --- |
| Ảnh train trong `train_img/` | 1835 |
| Ảnh validation/test trong `val2017/` | 552 |
| Tổng ảnh thật | 2387 |
| File JSON annotation/ground truth | 4 |
| Tổng file dữ liệu chính | 2391 |
| Dung lượng dữ liệu chính xấp xỉ | 1866.36 MB |

Tổng toàn bộ thư mục, bao gồm `__MACOSX` và `.DS_Store`:

| Loại file | Số lượng | Dung lượng |
| --- | --- | --- |
| `.png` | 2529 | 1865.04 MB |
| `.json` | 8 | 1.34 MB |
| `.DS_Store` | 6 | 0.02 MB |

Lưu ý: phần lớn file trong `__MACOSX` là AppleDouble metadata, không phải ảnh dữ liệu thật dù có đuôi `.png` hoặc `.json`.

## 2. Cấu trúc cây thư mục

```
KITTI-TS/
├── annotations/
│   ├── instances_test_a.json
│   └── instances_train_a.json
├── test/
│   ├── .DS_Store
│   └── sign_id_GT.json
├── train/
│   └── sign_id_GT.json
├── train_img/
│   ├── .DS_Store
│   └── *.png  (1835 ảnh)
├── val2017/
│   ├── .DS_Store
│   └── *.png  (552 ảnh)
└── __MACOSX/
    ├── annotations/
    ├── test/
    ├── train/
    ├── train_img/
    └── val2017/
```

## 3. Thống kê từng folder và vai trò

| Folder | File chính | Số lượng | Dung lượng xấp xỉ | Vai trò |
| --- | --- | --- | --- | --- |
| `annotations/` | `.json` | 2 | 0.78 MB | Annotation dạng COCO-like cho train và test/validation. |
| `train/` | `.json` | 1 | 0.45 MB | Ground truth theo từng `sign_id` cho tập train. |
| `test/` | `.json` | 1 | 0.11 MB | Ground truth theo từng `sign_id` cho tập test/validation. |
| `train_img/` | `.png` | 1835 | 1426.93 MB | Ảnh đầu vào cho tập train. |
| `val2017/` | `.png` | 552 | 438.09 MB | Ảnh đầu vào cho tập validation/test. |
| `__MACOSX/` | metadata | 149 file | rất nhỏ | Metadata sinh khi giải nén file zip từ macOS, thường bỏ qua khi train/test. |

Các file `.DS_Store` là file hệ thống của macOS, không có vai trò trong huấn luyện mô hình.

## 4. Chi tiết từng file dữ liệu chính

### `annotations/instances_train_a.json`

Vai trò: annotation dạng COCO-like cho tập train. File này liên kết ảnh trong `train_img/` với bounding box và class biển báo.

Thông tin chính:

| Trường | Ý nghĩa | Số lượng |
| --- | --- | --- |
| `images` | Danh sách ảnh train, gồm `file_name`, `height`, `width`, `id` | 1835 |
| `annotations` | Danh sách bounding box, category, image id | 2443 |
| `categories` | Danh sách class biển báo | 16 |
| `type` | Metadata loại annotation | 1 |

Ví dụ annotation đầu tiên:

```json
{"area":868,"iscrowd":0,"bbox":[331,128,31,28],"category_id":6,"ignore":0,"segmentation":[],"image_id":"2011_09_26_drive_0005_0000000008","id":1}
```

### `annotations/instances_test_a.json`

Vai trò: annotation dạng COCO-like cho tập test/validation. File này liên kết ảnh trong `val2017/` với bounding box và class biển báo.

Thông tin chính:

| Trường | Ý nghĩa | Số lượng |
| --- | --- | --- |
| `images` | Danh sách ảnh test/validation | 552 |
| `annotations` | Danh sách bounding box, category, image id | 619 |
| `categories` | Danh sách class biển báo | 16 |
| `type` | Metadata loại annotation | 1 |

Ví dụ annotation đầu tiên:

```json
{"area":1254,"iscrowd":0,"bbox":[378,86,22,57],"category_id":5,"ignore":0,"segmentation":[],"image_id":"2011_09_26_drive_0005_0000000077","id":1}
```

### `train/sign_id_GT.json`

Vai trò: ground truth cấp đối tượng biển báo cho tập train. Mỗi key lớn là một `sign_id`. Mỗi `sign_id` chứa class, hướng, vị trí GPS của biển báo, danh sách ảnh có biển đó, bbox theo từng ảnh, yaw và GPS camera theo từng ảnh.

Thông tin chính:

| Thành phần | Số lượng |
| --- | --- |
| Số đối tượng biển báo (`sign_id`) | 227 |
| Tổng số ảnh-tham-chiếu trong các `sign_id` | 2442 |
| Key trong mỗi sign object | `category`, `direction`, `images`, `Geolocation`, `image_yaws`, `image_geolocations` |

Ví dụ `sign_id` đầu tiên là `0`, thuộc category `5`, direction `1`, xuất hiện trong 27 ảnh.

### `test/sign_id_GT.json`

Vai trò: ground truth cấp đối tượng biển báo cho tập test/validation. Cấu trúc giống `train/sign_id_GT.json`.

Thông tin chính:

| Thành phần | Số lượng |
| --- | --- |
| Số đối tượng biển báo (`sign_id`) | 63 |
| Tổng số ảnh-tham-chiếu trong các `sign_id` | 619 |
| Key trong mỗi sign object | `category`, `direction`, `images`, `Geolocation`, `image_yaws`, `image_geolocations` |

Ví dụ `sign_id` đầu tiên là `2`, thuộc category `4`, direction `2`, xuất hiện trong 8 ảnh.

### `train_img/*.png`

Vai trò: ảnh RGB đầu vào của tập train. Tên file theo pattern:

```
YYYY_MM_DD_drive_XXXX_FRAME.png
```

Ví dụ 5 file đầu:

```
2011_09_26_drive_0005_0000000008.png
2011_09_26_drive_0005_0000000010.png
2011_09_26_drive_0005_0000000012.png
2011_09_26_drive_0005_0000000014.png
2011_09_26_drive_0005_0000000016.png
```

### `val2017/*.png`

Vai trò: ảnh RGB đầu vào của tập validation/test. Tên file cùng pattern với `train_img/`.

Ví dụ 5 file đầu:

```
2011_09_26_drive_0005_0000000077.png
2011_09_26_drive_0005_0000000078.png
2011_09_26_drive_0005_0000000079.png
2011_09_26_drive_0005_0000000080.png
2011_09_26_drive_0005_0000000081.png
```

## 5. Classes biển báo

Hai file `instances_*_a.json` dùng chung 16 class:

| ID | Name |
| --- | --- |
| 1 | `SpeedLimit30` |
| 2 | `SpeedLimitOff` |
| 3 | `SpeedLimit60` |
| 4 | `RoadWork` |
| 5 | `Yield` |
| 6 | `MainRoad` |
| 7 | `NoStop` |
| 8 | `NoStay` |
| 9 | `EndOfRoad` |
| 10 | `OneWayRight` |
| 11 | `OneWayLeft` |
| 12 | `NoRoad` |
| 13 | `Left` |
| 14 | `PriorotyRoad` |
| 15 | `PedestrianCross` |
| 16 | `PassOnRight` |

Lưu ý: class `PriorotyRoad` có vẻ bị sai chính tả so với `PriorityRoad`, nhưng report giữ nguyên tên trong file.

## 6. Phân bố annotation theo class

| Category ID | Train count | Test/val count |
| --- | --- | --- |
| 1 | 148 | 54 |
| 2 | 76 | 14 |
| 3 | 46 | 27 |
| 4 | 41 | 12 |
| 5 | 334 | 67 |
| 6 | 230 | 80 |
| 7 | 241 | 60 |
| 8 | 117 | 41 |
| 9 | 108 | 18 |
| 10 | 149 | 30 |
| 11 | 159 | 40 |
| 12 | 289 | 57 |
| 13 | 59 | 10 |
| 14 | 134 | 33 |
| 15 | 170 | 24 |
| 16 | 142 | 52 |

## 7. Kích thước ảnh theo annotation

Các ảnh không hoàn toàn cùng resolution. Thống kê từ trường `width` và `height` trong JSON:

| Resolution | Train images | Test/val images |
| --- | --- | --- |
| `1224x370` | 29 | 18 |
| `1226x370` | 535 | 205 |
| `1238x374` | 16 | 22 |
| `1241x376` | 607 | 82 |
| `1242x375` | 648 | 225 |

## 8. 5 dòng đầu của các file JSON

Các file JSON gốc đang được minify thành một dòng rất dài. Để dễ đọc, phần dưới đây là 5 dòng đầu sau khi format lại JSON.

### `annotations/instances_test_a.json`

```json
{
  "images": [
    {
      "file_name": "2011_09_26_drive_0005_0000000077.png",
      "height": 375,
```

### `annotations/instances_train_a.json`

```json
{
  "images": [
    {
      "file_name": "2011_09_26_drive_0005_0000000008.png",
      "height": 375,
```

### `test/sign_id_GT.json`

```json
{
  "2": {
    "category": "4",
    "direction": "2",
    "images": {
```

### `train/sign_id_GT.json`

```json
{
  "0": {
    "category": "5",
    "direction": "1",
    "images": {
```

## 9. Nhận xét sử dụng dataset

- Với bài toán object detection, nên dùng `annotations/instances_train_a.json` cùng `train_img/` để train và `annotations/instances_test_a.json` cùng `val2017/` để đánh giá.
- Với bài toán tracking/định danh biển báo qua nhiều frame, các file `sign_id_GT.json` hữu ích hơn vì gom các bbox theo cùng một `sign_id`.
- Nên bỏ qua `__MACOSX/` và `.DS_Store` khi viết dataloader.
- `annotations_train_a.json` có 2443 annotation, trong khi tổng image refs trong `train/sign_id_GT.json` là 2442. Có thể có 1 annotation không map trực tiếp vào `sign_id_GT`, nên nếu cần đối chiếu tuyệt đối thì nên kiểm tra thêm bằng script.