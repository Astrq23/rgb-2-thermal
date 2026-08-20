# Hướng dẫn sử dụng

Quy trình vận hành đầy đủ, từ tài khoản Kaggle trắng đến ảnh nhiệt sinh ra.

[`README.md`](../README.md) giải thích dự án **làm gì** và **thiết kế thế nào**.
Tài liệu này chỉ nói **làm theo thứ tự nào**.

---

## Mục lục

- [0. Chuẩn bị tài khoản](#0-chuẩn-bị-tài-khoản-làm-một-lần)
- [1. Chọn đường đi](#1-chọn-đường-đi)
- [2. Nạp code vào Kaggle](#2-nạp-code-vào-kaggle)
- [3. Attach dataset](#3-attach-dataset)
- [4. Kiểm tra dataset](#4-kiểm-tra-dataset--bước-không-được-bỏ-qua)
- [5. Kiểm tra căn chỉnh](#5-kiểm-tra-căn-chỉnh--bước-thứ-hai-không-được-bỏ-qua)
- [6. Xuất subset](#6-xuất-subset-khuyến-nghị-mạnh)
- [7. Train](#7-train)
- [8. Vượt giới hạn 12 giờ](#8-vượt-giới-hạn-12-giờ)
- [9. Đánh giá](#9-đánh-giá)
- [10. Sinh ảnh nhiệt](#10-sinh-ảnh-nhiệt-từ-ảnh-của-bạn)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Tùy chỉnh thường dùng](#tùy-chỉnh-thường-dùng)
- [Chạy ở máy local](#chạy-ở-máy-local)

---

## 0. Chuẩn bị tài khoản (làm một lần)

**Settings → Phone Verification**, xác thực số điện thoại.

Không có bước này thì **không bật được Internet** (notebook không `git clone` được
code) và **không dùng được GPU**. Đây là nguyên nhân số một khiến người mới bị kẹt
ngay cell đầu tiên.

Sau khi xác thực: ~30 giờ GPU/tuần, mỗi session tối đa 12 giờ.

---

## 1. Chọn đường đi

| | Đường đầy đủ | Đường nhanh |
|---|---|---|
| Attach | 4 dataset gốc, hàng chục GB | 1 subset tự tạo, vài trăm MB |
| Thời gian khởi động session | Lâu, lặp lại mỗi lần | Vài giây |
| Khi nào dùng | **Lần đầu tiên**, bắt buộc | Mọi lần sau |

Lần đầu buộc phải đi đường đầy đủ — bạn cần dataset gốc để tạo ra subset. Sau khi
làm [bước 6](#6-xuất-subset-khuyến-nghị-mạnh) một lần, những lần sau chỉ đi đường nhanh.

---

## 2. Nạp code vào Kaggle

[kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** →
**File → Import Notebook → Link**:

```
https://raw.githubusercontent.com/Astrq23/rgb-2-thermal/main/notebooks/00_explore_datasets.ipynb
```

Nếu import không nhận, tải file `.ipynb` từ GitHub về rồi **Import → Upload**.

**Settings** (panel phải):
- Accelerator: `GPU P100` hoặc `GPU T4 x2`
- Internet: `On`

> Notebook clone nhánh `main`. Muốn chạy code chưa release, sửa `BRANCH = "dev"`
> trong cell setup — đừng push thẳng vào `main`.

Notebook chỉ điều phối; toàn bộ code được clone từ GitHub. Sửa code thì sửa trong
repo rồi chạy lại cell setup, **không dán code vào notebook**.

---

## 3. Attach dataset

Panel phải → **Add Data** → **Datasets**, tìm theo slug:

| Slug | Vai trò | Bắt buộc? |
|---|---|---|
| `brendanalvey/visdrone-dronevehicle` | Nguồn UAV chính | **Có** |
| `monishshrivastava1/llvip-dataset` | Cảnh đêm, người đi bộ | Nên có |
| `samdazel/teledyne-flir-adas-thermal-dataset-v2` | Bối cảnh mặt đất | Không |
| `pandrii000/hituav-a-highaltitude-infrared-thermal-dataset` | Đánh giá FID | Không |

Attach thiếu vẫn chạy được — dataset không tìm thấy sẽ bị bỏ qua, không lỗi.

Nếu Kaggle báo vượt giới hạn dung lượng, **bỏ FLIR trước**: nó nặng nhất, căn chỉnh
kém nhất, và chỉ chiếm 10% trọng số lấy mẫu.

---

## 4. Kiểm tra dataset — bước không được bỏ qua

Chạy notebook `00_explore_datasets.ipynb` tuần tự. Cell `inspect_datasets.py` in
cây thư mục thật của `/kaggle/input` và kết quả dò của từng adapter.

**Vì sao bắt buộc:** các bản mirror trên Kaggle do cộng đồng upload, đôi khi tổ chức
thư mục khác bản gốc. Adapter được viết theo layout gốc, nên đây là chỗ duy nhất phát
hiện sai lệch trước khi mọi thứ phía sau hỏng theo.

Đọc phần **ADAPTER PROBE**:

| Kết quả | Nghĩa | Xử lý |
|---|---|---|
| `OK — N pairs` | Đúng | Đi tiếp |
| `NOT MOUNTED` | Chưa attach, hoặc slug khác | Attach, hoặc thêm path vào `search_roots` |
| `root exists but 0 pairs matched` | Layout khác dự đoán | Sửa YAML, xem bên dưới |

**Số cặp kỳ vọng** — lệch nhiều nghĩa là adapter chỉ thấy một phần dữ liệu:

| Dataset | Kỳ vọng |
|---|---|
| DroneVehicle | ~28.400 cặp |
| LLVIP | ~15.400 cặp |
| FLIR v2 | vài nghìn cặp |
| HIT-UAV | ~2.900 ảnh (thermal-only, đúng thiết kế) |

### Khi gặp `0 pairs matched`

So cây thư mục thật với `configs/datasets/<tên>.yaml` rồi sửa **config, không sửa Python**:

| Tình huống | Sửa gì |
|---|---|
| Thư mục RGB/thermal tên lạ | `thermal_tokens` / `rgb_tokens` |
| Hai modality khác tên file | `key_regex` |
| Không có token modality nào | `force_modality: thermal` |
| Đường dẫn mount khác | `search_roots` |

Sửa xong: commit lên nhánh, tạo MR, merge, rồi chạy lại cell setup (nó `git pull`).

---

## 5. Kiểm tra căn chỉnh — bước thứ hai không được bỏ qua

Cell `check_alignment.py` xuất ảnh 4 panel cho mỗi dataset. **Nhìn panel thứ 4**:
biên nhiệt vẽ màu đỏ chồng lên ảnh RGB.

- Biên nằm đúng trên xe, nhà, vạch đường → căn chỉnh tốt, giữ dataset.
- Biên trôi lệch khỏi cấu trúc → dataset lệch.

**Vì sao quan trọng:** ảnh lệch **không gây lỗi**. Loss vẫn giảm đẹp, biểu đồ vẫn
khỏe mạnh, model chỉ âm thầm học cách làm mờ. Không có chỉ số nào cảnh báo bạn.

DroneVehicle và LLVIP căn chỉnh bằng phần cứng nên gần như chắc chắn tốt.
**FLIR thì không** — hai camera khác FOV, và `fov_crop: 0.62` trong config là ước
lượng, chưa phải hiệu chuẩn. Quét thử vài giá trị:

```bash
python scripts/check_alignment.py --dataset flir_v2 --fov-crop 0.55 --n 6
python scripts/check_alignment.py --dataset flir_v2 --fov-crop 0.70 --n 6
```

Ghi giá trị tốt nhất vào `configs/datasets/flir_v2.yaml`. Nếu không giá trị nào khớp,
đặt `enabled: false` — **ít dữ liệu tốt hơn dữ liệu dạy model làm mờ.**

---

## 6. Xuất subset (khuyến nghị mạnh)

Phần lớn dung lượng dataset gốc là **độ phân giải bị vứt đi ngay**: DroneVehicle gửi
ảnh 840×712, train cắt viền rồi resize xuống 256×256. Bạn trả giá attach cho độ phân
giải đó mỗi session, rồi lại decode nó ở mỗi epoch.

Chạy **một lần**, khi dataset gốc còn đang attach:

```bash
python scripts/export_subset.py \
    --manifest /kaggle/working/manifest.csv \
    --out /kaggle/working/subset \
    --per-dataset 8000
```

Rồi:

1. **Save Version → Save & Run All**, đợi chạy xong.
2. Từ run đã xong: **Notebook Output → Create Dataset**, đặt tên `rgb-thermal-subset`
   (đúng tên này thì `search_roots` mặc định tìm thấy ngay).
3. Những session sau chỉ attach dataset đó, rồi:

```bash
python scripts/build_manifest.py \
    --dataset-specs 'configs/datasets/subset*.yaml' \
    --out /kaggle/working/manifest.csv
```

> Attach subset **hoặc** dataset gốc, **đừng cả hai** — mỗi mẫu sẽ vào manifest hai lần.

Bản subset giữ nguyên tên dataset gốc, trọng số lấy mẫu, bảng chỉ số tách theo dataset
và nhóm scene chống rò rỉ split — tất cả được mã hoá trong đường dẫn xuất ra.
Dữ liệu thermal-only đi vào `reference/` và vẫn dùng cho FID không cặp.

---

## 7. Train

Import `01_kaggle_train.ipynb`, attach dataset và bật GPU + Internet như trên.

### 7.1. Smoke run trước (~2 phút)

```bash
python scripts/train.py --config configs/smoke.yaml \
    --manifest /kaggle/working/manifest.csv
```

50 step. Nó chạy qua manifest, DataLoader, cả hai network, loss, AMP, checkpoint và
xuất ảnh. Ảnh ra sẽ là nhiễu — điều cần kiểm tra là lưới ảnh **có render được không**,
đúng 3 cột và đúng số hàng.

Hai phút ở đây đáng giá hơn nhiều so với việc phát hiện một lỗi config ở giờ thứ chín.

### 7.2. Train đầy đủ

```bash
python scripts/train.py \
    --config configs/pix2pix_uav.yaml \
    --manifest /kaggle/working/manifest.csv \
    --resume auto \
    --set train.max_hours=11
```

`max_hours=11` là biên an toàn: run tự dừng và checkpoint ở giờ 11 thay vì bị Kaggle
giết giữa epoch ở giờ 12.

Sau **epoch đầu tiên**, trainer in throughput đo được (img/s) và ETA cho số epoch còn
lại. Đó là lúc quyết định có cần rút ngắn run không — bằng số thật, không phải đoán.

### 7.3. Đọc biểu đồ

| Dấu hiệu | Nghĩa | Xử lý |
|---|---|---|
| `train_g_l1` giảm, `val_l1` bám theo | Bình thường | — |
| Khoảng cách train/val ngày càng xa | Overfitting | Giảm `train.epochs`, hoặc thêm dữ liệu |
| `train_d_total` → ~0 | Discriminator thắng, generator hết gradient | Giảm `train.lr`, hoặc tăng `train.lambda_l1` |
| `train_g_l1` phẳng ngay từ đầu | Vấn đề ở **dữ liệu**, không phải model | Quay lại [bước 5](#5-kiểm-tra-căn-chỉnh--bước-thứ-hai-không-được-bỏ-qua) |
| Loss adversarial dao động | Bình thường với GAN | — |

Đánh giá ảnh sinh ra bằng **tính hợp lý về nhiệt**, không phải khớp từng pixel: xe và
người phải sáng hơn mặt đường và cây cối; khoang máy nóng hơn thân xe; nhà giữ nhiệt
sau khi trời tối. Thứ tự đó đúng quan trọng hơn khớp chính xác một frame.

---

## 8. Vượt giới hạn 12 giờ

Session bị kill ở mốc 12 giờ, không báo trước.

1. **Save Version → Save & Run All**, đợi chạy xong.
2. Notebook mới → **Add Data → Notebook Output** → chọn run vừa rồi.
3. Copy checkpoint rồi chạy tiếp:

```bash
mkdir -p /kaggle/working/outputs/pix2pix_uav/checkpoints
cp /kaggle/input/<notebook-output>/outputs/pix2pix_uav/checkpoints/last.pt \
   /kaggle/working/outputs/pix2pix_uav/checkpoints/

python scripts/train.py --config configs/pix2pix_uav.yaml \
    --manifest /kaggle/working/manifest.csv --resume auto
```

Checkpoint mang đủ hai network, hai optimizer, hai scheduler, AMP scaler và RNG state
nên đường loss nối liền, không gãy khúc.

> **Tải `best.pt` về máy trước khi session kết thúc.** `/kaggle/working` không được giữ lại.

---

## 9. Đánh giá

```bash
python scripts/evaluate.py \
    --checkpoint /kaggle/working/outputs/pix2pix_uav/checkpoints/best.pt \
    --config configs/pix2pix_uav.yaml \
    --manifest /kaggle/working/manifest.csv \
    --split test
```

Chỉ số được tách **theo từng dataset**. Đọc riêng dòng `dronevehicle` — con số gộp sẽ
bị các frame FLIR mặt đất (vốn dễ hơn) kéo lên và che mất việc domain UAV có thật sự
tốt lên hay không.

| Chỉ số | Trả lời | Hướng tốt |
|---|---|---|
| PSNR | Sát frame ground-truth này đến đâu | Cao |
| SSIM | Giống về cấu trúc | Cao |
| LPIPS | Khoảng cách cảm nhận — thường hữu ích nhất | Thấp |
| FID | Độ thật về phân phối | Thấp |
| FID không cặp vs HIT-UAV | Có giống ảnh nhiệt UAV thật không | Thấp |

Một vệt nhiệt hợp lý nhưng lệch vài pixel bị PSNR phạt nặng hơn một ảnh mờ nhưng
"đúng chỗ" — trong khi ảnh đầu mới hữu ích hơn. Vì vậy phải đọc LPIPS và FID cùng lúc.

---

## 10. Sinh ảnh nhiệt từ ảnh của bạn

Notebook `02_kaggle_inference.ipynb`, sửa `CHECKPOINT` và `INPUT_DIR` ở cell đầu.

```bash
python scripts/predict.py \
    --checkpoint <đường dẫn best.pt> \
    --input <thư mục ảnh> \
    --out /kaggle/working/predictions \
    --colormap inferno \
    --side-by-side
```

| Cờ | Tác dụng |
|---|---|
| *(không có)* | Ảnh xám 1 kênh — dùng khi output đưa vào model khác |
| `--colormap inferno` | Tô màu để người xem |
| `--side-by-side` | Ảnh gốc cạnh ảnh sinh ra |
| `--keep-size` | Phóng output 256×256 về đúng độ phân giải ảnh vào |
| `--domain dronevehicle` | Chọn "phong cách sensor" (chỉ khi train với `use_domain_embedding: true`) |

> **Model sinh hình thái nhiệt, không phải nhiệt độ.** Pixel 200 nghĩa là "nóng so với
> chính frame này", không bao giờ là "44 °C". Chi tiết ở phần Giới hạn trong
> [`README.md`](../README.md).

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `git clone` treo / lỗi mạng | Internet chưa bật | Xác thực SĐT, bật Internet |
| `ModuleNotFoundError: rgb2thermal` | Cell `pip install -e .` chưa chạy | Chạy lại cell setup |
| `Manifest is EMPTY` | Chưa attach dataset nào | `python scripts/inspect_datasets.py` |
| `0 pairs matched` | Layout mirror khác dự đoán | Sửa `configs/datasets/*.yaml`, xem [bước 4](#khi-gặp-0-pairs-matched) |
| `No training rows in the manifest` | Manifest có nhưng không có split `train` | Build lại manifest |
| CUDA out of memory | Batch quá lớn | `--set train.batch_size=8` |
| GPU rảnh, DataLoader chậm | Nút thắt ở decode ảnh | `--set data.num_workers=4`, hoặc dùng subset |
| Ảnh sinh ra xám phẳng | Thường do dataset lệch | Quay lại [bước 5](#5-kiểm-tra-căn-chỉnh--bước-thứ-hai-không-được-bỏ-qua) |
| `Unknown key(s) in config section` | Gõ sai tên tham số | Đọc tên đúng trong thông báo lỗi |
| `WARNING: scenes span multiple splits` | Split gốc của dataset cắt ngang nhóm scene | Bình thường; muốn chặt hơn thì `--set data.respect_split_hint=false` |
| Session bị kill giữa chừng | Chạm mốc 12 giờ | Đặt `train.max_hours=11`, xem [bước 8](#8-vượt-giới-hạn-12-giờ) |
| LPIPS/FID bị bỏ qua | Không tải được trọng số | Bật Internet, hoặc bỏ chúng khỏi `eval.metrics` |

---

## Tùy chỉnh thường dùng

Mọi giá trị trong config đều override được từ dòng lệnh bằng `--set`:

```bash
python scripts/train.py --config configs/pix2pix_uav.yaml \
    --set train.epochs=60 train.batch_size=8 data.num_workers=4
```

| Muốn gì | Lệnh |
|---|---|
| Kết quả sớm | `--set train.epochs=10` |
| Dùng T4 x2 | `--set train.batch_size=24 data.num_workers=4` |
| Hết VRAM | `--set train.batch_size=8` |
| Train nhanh, chấp nhận ảnh nhỏ | `--set data.image_size=128 model.n_down=7` |
| Chỉ train trên UAV | `--set 'data.sample_weights={dronevehicle: 1.0}'` |
| Baseline U-Net (không GAN) | `--set model.name=unet` |
| Bật conditioning theo sensor | `--set model.use_domain_embedding=true` |
| Split chặt, không theo split gốc | `--set data.respect_split_hint=false` |

Gõ sai tên tham số sẽ báo lỗi ngay thay vì bị bỏ qua âm thầm — một `epocs: 100` không
được phát hiện sẽ đốt nguyên một session GPU.

---

## Chạy ở máy local

Không cần GPU, không cần tải dataset. Test dùng cây thư mục giả dựng bằng PIL, tái
hiện đúng đặc thù của từng dataset thật.

```bash
git clone https://github.com/Astrq23/rgb-2-thermal.git
cd rgb-2-thermal
pip install -e ".[dev,train]"
pytest -q
```

Sau khi sửa nội dung notebook:

```bash
python scripts/build_notebooks.py
```

Quy trình đóng góp code (nhánh, merge request): xem [`CLAUDE.md`](../CLAUDE.md).
