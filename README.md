# RGB → Thermal cho ảnh UAV

Sinh ảnh nhiệt (thermal/infrared) từ ảnh RGB, tập trung vào góc nhìn từ drone.
Huấn luyện bằng Pix2Pix trên **nhiều dataset Kaggle được gộp và chuẩn hóa** về
một schema chung. Thiết kế để chạy trọn vẹn trong một session Kaggle.

> Tài liệu này viết bằng tiếng Việt; code, docstring và log trong repo dùng tiếng Anh.

---

## Chạy trên Kaggle trong 3 bước

1. Tạo notebook mới → **Add data** → attach các dataset ở [bảng bên dưới](#dataset).
2. **Settings** → Accelerator: **GPU**, Internet: **ON** (cần để `git clone`).
3. Mở [`notebooks/01_kaggle_train.ipynb`](notebooks/01_kaggle_train.ipynb) và chạy tuần tự.

Ba notebook, chạy theo thứ tự:

| Notebook | Việc |
|---|---|
| [`00_explore_datasets.ipynb`](notebooks/00_explore_datasets.ipynb) | Kiểm tra layout thật của dataset + kiểm tra căn chỉnh RGB/thermal |
| [`01_kaggle_train.ipynb`](notebooks/01_kaggle_train.ipynb) | Smoke run → train đầy đủ → biểu đồ → đánh giá |
| [`02_kaggle_inference.ipynb`](notebooks/02_kaggle_inference.ipynb) | Sinh ảnh nhiệt từ ảnh RGB bất kỳ |

Notebook chỉ điều phối — toàn bộ code được `git clone` từ repo này. Sửa code thì
sửa trong repo rồi chạy lại cell setup, **không dán code vào notebook**.

---

## Dataset

| Dataset | Slug Kaggle | Vai trò | Góc nhìn |
|---|---|---|---|
| **DroneVehicle** | `brendanalvey/visdrone-dronevehicle` | Nguồn chính (weight 0.60) | UAV nhìn xuống, ngày + đêm |
| **LLVIP** | `monishshrivastava1/llvip-dataset` | Cảnh đêm, người đi bộ (0.30) | Camera gắn cao |
| **FLIR ADAS v2** | `samdazel/teledyne-flir-adas-thermal-dataset-v2` | Đa dạng bối cảnh (0.10) | Ngang tầm mắt |
| **HIT-UAV** | `pandrii000/hituav-a-highaltitude-infrared-thermal-dataset` | **Chỉ đánh giá** | UAV độ cao lớn |

Attach thiếu một vài dataset vẫn chạy được — dataset không có sẽ bị bỏ qua, không lỗi.

Chi tiết về license và cách trích dẫn: [`DATASETS.md`](DATASETS.md).

**HIT-UAV không có ảnh RGB ghép cặp**, nên không dùng để train paired được. Nó
đóng vai trò tập tham chiếu để tính **FID không cặp**: trả lời câu hỏi mà PSNR
không trả lời được — "ảnh sinh ra có *giống ảnh nhiệt UAV thật* không?", chứ
không chỉ "có khớp đúng frame ground-truth này không?".

---

## Vấn đề cốt lõi: chuẩn hóa dữ liệu

Bốn dataset tổ chức hoàn toàn khác nhau:

```
DroneVehicle   train/trainimg/00001.jpg   ↔  train/trainimgr/00001.jpg
LLVIP          visible/train/010001.jpg   ↔  infrared/train/010001.jpg
FLIR v2        images_rgb_train/data/video-A-frame-000001-<hash1>.jpg
               images_thermal_train/data/video-A-frame-000001-<hash2>.jpg
HIT-UAV        normal/images/train/0001.jpg          (không có RGB)
```

Pipeline xử lý qua 4 tầng, mỗi tầng test độc lập được:

### 1. Adapter — tự dò cặp ảnh

Không hard-code từng layout. Mỗi thành phần trong đường dẫn được phân loại bằng
**modality token**, rồi token đó bị thay bằng placeholder để tạo *pairing key*:

```
train/trainimg/00001.jpg   →  train/<M>/00001
train/trainimgr/00001.jpg  →  train/<M>/00001   ← cùng key → ghép cặp
```

Token chia hai tầng. Token mạnh (`thermal`, `rgb`, `infrared`, `imgr`…) không thể
nhầm. Token yếu (`img`, `ir`) chỉ được xét khi mọi token mạnh đều trượt — nếu
không, `images_rgb_train` sẽ khớp `img` ở vị trí 0 và không bao giờ tới được
`rgb` mới là phần mang ý nghĩa.

Trường hợp adapter không xử lý được bằng khai báo:

| Tình huống | Giải pháp (trong YAML) |
|---|---|
| Thư mục đặt tên lạ | `thermal_tokens` / `rgb_tokens` |
| Hai modality khác tên file (FLIR) | `key_regex` |
| Không có token modality nào (HIT-UAV) | `force_modality: thermal` |

> **Nếu mirror trên Kaggle có layout khác dự đoán**, chạy
> `python scripts/inspect_datasets.py` để in cây thư mục thật, rồi sửa
> `configs/datasets/*.yaml`. Gần như mọi trường hợp chỉ cần sửa config,
> không cần sửa Python.

### 2. Manifest — một bảng duy nhất

`scripts/build_manifest.py` gộp tất cả thành `manifest.csv`:

```
rgb_path, thermal_path, dataset, viewpoint, time_of_day, group_key, split_hint, split
```

Từ đây trở đi không còn thành phần nào biết DroneVehicle khác LLVIP.

**Chia split theo `group_key`, không random theo ảnh.** Các dataset này là frame
video liên tiếp — hai frame cạnh nhau gần như giống hệt. Chia ngẫu nhiên sẽ đẩy
ảnh gần trùng vào cả train lẫn val và làm mọi chỉ số đẹp giả. Việc chia dùng
hash MD5 chứ không dùng `random`, nên split giữ nguyên khi resume session Kaggle.

### 3. Chuẩn hóa pixel

- **Hình học** — DroneVehicle cắt viền trắng 100px (840×712 → 640×512);
  FLIR center-crop theo tỉ lệ FOV.
- **Cường độ** — ép thermal về 1 kênh, rồi **percentile stretch 1–99% cho từng ảnh**.

Bước stretch chính là thứ khiến việc gộp ba camera nhiệt khác nhau trở nên có
nghĩa: nó triệt tiêu khác biệt gain/offset giữa các sensor. Đánh đổi được nói rõ
ở phần [Giới hạn](#giới-hạn).

### 4. Trộn có trọng số

Nếu chỉ nối các dataset rồi shuffle thì tỉ lệ trộn sẽ do **kích thước file**
quyết định. Ở đây mỗi dataset có xác suất mục tiêu khai báo trong config, và
trọng số từng mẫu được đặt bằng `w_d / n_d` để xác suất rút trúng dataset `d`
đúng bằng `w_d`, bất kể nó đóng góp bao nhiêu ảnh.

---

## Model

**Pix2Pix**: U-Net generator + PatchGAN discriminator có điều kiện.

- Generator: U-Net 8 tầng, skip connection, ra 1 kênh + `tanh`.
- Discriminator: PatchGAN 70×70, nhận `concat(rgb, thermal)` → phán xét từng vùng.
- Loss: `L_GAN + 100 · L1`. L1 lo cấu trúc; adversarial lo chi tiết tần số cao mà
  L1 một mình luôn làm mờ.
- Adam lr 2e-4, decay tuyến tính nửa sau. AMP bật mặc định.

Tùy chọn `model.use_domain_embedding` (mặc định tắt): nhúng FiLM theo dataset tại
bottleneck. Khi gộp nhiều camera, một cảnh RGB có nhiều bản thermal hợp lệ khác
nhau — conditioning cho phép một model biểu diễn cả ba thay vì lấy trung bình.

Baseline U-Net thuần (chỉ L1, không GAN) dùng chung khung train:
`--set model.name=unet`.

---

## Cấu trúc thư mục

```
configs/
  base.yaml                 defaults; các config khác `extends:` từ đây
  pix2pix_uav.yaml          config train chính
  smoke.yaml                chạy thử 50 step
  datasets/*.yaml           mô tả khai báo cho từng dataset
src/rgb2thermal/
  config.py                 load YAML + override từ CLI
  data/
    adapters/               modality.py, base.py, flir_adas_v2.py, registry.py
    manifest.py             gộp + chia split
    normalize.py            crop viền, percentile stretch  (chỉ PIL/numpy)
    dataset.py, sampler.py  Dataset + weighted sampler
  models/                   unet_generator.py, patchgan.py, build.py
  engine/                   trainer.py, evaluator.py, amp.py
  losses.py  metrics.py  viz.py  utils/
scripts/
  inspect_datasets.py       CHẠY ĐẦU TIÊN trên Kaggle
  build_manifest.py  check_alignment.py
  train.py  evaluate.py  predict.py
  build_notebooks.py        sinh notebook từ source
notebooks/                  3 notebook Kaggle
tests/                      chạy được mà không cần tải dataset
```

---

## Các lệnh thường dùng

```bash
python scripts/inspect_datasets.py                      # xem gì đang được mount
python scripts/build_manifest.py --out manifest.csv     # gộp dataset
python scripts/check_alignment.py --dataset flir_v2     # kiểm tra căn chỉnh
python scripts/train.py --config configs/smoke.yaml     # smoke run ~2 phút
python scripts/train.py --config configs/pix2pix_uav.yaml --resume auto
python scripts/evaluate.py --checkpoint outputs/pix2pix_uav/checkpoints/best.pt
python scripts/predict.py --checkpoint ... --input anh/ --colormap inferno
```

Mọi giá trị config đều override được từ dòng lệnh:

```bash
python scripts/train.py --config configs/pix2pix_uav.yaml \
    --set train.epochs=60 train.batch_size=8 data.num_workers=4
```

Gõ sai tên key sẽ báo lỗi ngay thay vì bị bỏ qua âm thầm — một `epocs: 100` không
được phát hiện sẽ đốt nguyên một session GPU.

---

## Chạy quá 12 giờ của Kaggle

Session Kaggle bị kill ở mốc 12 giờ, không báo trước. Checkpoint được lưu sau mỗi
epoch kèm **đủ trạng thái**: hai network, hai optimizer, hai scheduler, AMP scaler,
số epoch và RNG state — nên khi resume, đường loss không bị gãy khúc.

1. **Save Version → Save & Run All**, đợi chạy xong.
2. Notebook mới → **Add data → Notebook Output** → chọn run vừa rồi.
3. Copy `last.pt` vào `/kaggle/working/outputs/pix2pix_uav/checkpoints/` rồi chạy
   `python scripts/train.py --config configs/pix2pix_uav.yaml --resume auto`.

Nhớ tải `best.pt` về máy trước khi session kết thúc — `/kaggle/working` không được giữ lại.

---

## Đánh giá

Chỉ số được báo cáo **tách riêng theo từng dataset**, không chỉ số gộp. Một con số
trung bình sẽ che mất điều quan trọng nhất: domain UAV có thật sự tốt lên không,
hay cải thiện chỉ đến từ các frame FLIR mặt đất vốn dễ hơn.

| Chỉ số | Trả lời câu hỏi |
|---|---|
| PSNR / SSIM | Sát với đúng frame ground-truth này đến đâu |
| LPIPS | Khoảng cách cảm nhận — thường là chỉ số hữu ích nhất |
| FID | Độ thật về mặt phân phối |
| **FID không cặp vs HIT-UAV** | Có giống ảnh nhiệt UAV thật không |

---

## Thêm một dataset mới

Tạo một file YAML trong `configs/datasets/`. Thường không cần viết Python:

```yaml
name: my_dataset
adapter: generic
viewpoint: uav_mid
sample_weight: 0.15
search_roots:
  - /kaggle/input/my-dataset
  - /kaggle/input/*my*dataset*
preprocess:
  rgb: {crop_border: 0}
  thermal: {invert: false}
```

Chạy `python scripts/inspect_datasets.py --dataset my_dataset` để kiểm tra adapter
có tìm đúng cặp không, rồi build lại manifest.

---

## Giới hạn

**Model sinh ra *hình thái nhiệt*, không phải nhiệt độ.** Bước percentile stretch
cho từng ảnh — thứ khiến ba camera nhiệt khác nhau gộp lại được — đồng thời loại
bỏ giá trị radiometric tuyệt đối. Pixel giá trị 200 nghĩa là "nóng so với chính
frame này", không bao giờ là "44 °C".

Với bài toán UAV, đánh đổi này thường chấp nhận được: tăng cường dữ liệu cho
detection/segmentation, xem trước payload nhiệt sẽ thấy gì, dựng thử pipeline
trước khi có phần cứng. Nhưng nó **không thay thế được sensor radiometric**, và
không thành phần nào phía sau được phép coi nó như vậy.

**FLIR ADAS căn chỉnh không chặt.** Hai camera khác FOV. Giá trị
`preprocess.rgb.fov_crop` trong config là ước lượng, chưa phải hiệu chuẩn. Chạy
`scripts/check_alignment.py` và xem panel thứ 4 trước khi tin dataset này — ảnh
lệch không gây lỗi, nó chỉ âm thầm dạy model làm mờ.

**Chỉ số paired có giới hạn ý nghĩa.** Một vệt nhiệt hợp lý nhưng lệch vài pixel
sẽ bị PSNR phạt nặng hơn một ảnh mờ nhưng "đúng chỗ", trong khi ảnh đầu mới là
ảnh hữu ích hơn. Vì vậy cần đọc LPIPS và FID cùng lúc.

---

## Phát triển ở máy local

Không cần GPU và không cần tải dataset — test dùng cây thư mục giả dựng bằng PIL,
tái hiện đúng đặc thù của từng dataset thật.

```bash
pip install -e ".[dev,train]"
pytest -q
```

Test bao phủ: phân loại modality, ghép cặp của cả 4 layout, cắt viền, percentile
stretch xuyên camera, chống rò rỉ split, shape của model, một bước train đầy đủ
kèm checkpoint + resume, và tính hợp lệ của notebook.

Sau khi sửa nội dung notebook, chạy lại:

```bash
python scripts/build_notebooks.py
```

---

## License

Code: MIT — xem [`LICENSE`](LICENSE).
Dataset có license riêng, phần lớn **phi thương mại / nghiên cứu**: xem [`DATASETS.md`](DATASETS.md).
