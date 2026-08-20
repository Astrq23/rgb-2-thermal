# Nguồn dữ liệu, license và cách trích dẫn

Repo này **không chứa và không phân phối lại** bất kỳ ảnh nào. Toàn bộ dữ liệu
được attach trực tiếp từ Kaggle trong session, dưới license của chính chủ sở hữu.

Phần lớn các dataset dưới đây giới hạn **nghiên cứu / phi thương mại**. Nếu dự án
của bạn có yếu tố thương mại, hãy đọc license gốc trước khi dùng.

---

## 1. DroneVehicle — nguồn UAV chính

| | |
|---|---|
| Kaggle | [`brendanalvey/visdrone-dronevehicle`](https://www.kaggle.com/datasets/brendanalvey/visdrone-dronevehicle) |
| Gốc | https://github.com/VisDrone/DroneVehicle |
| Quy mô | ~28.439 cặp RGB/IR (56.878 ảnh) |
| Kích thước | 840×712 khi tải về; **640×512 sau khi cắt viền trắng 100px** |
| Góc nhìn | UAV nhìn xuống, ngày và đêm |
| License | Nghiên cứu / phi thương mại (theo điều khoản VisDrone) |

Ảnh gốc được chèn viền trắng 100px mỗi cạnh để nhóm tác giả annotate được vật thể
sát biên. Config `dronevehicle.yaml` cắt viền này; nếu bỏ qua, model sẽ tốn dung
lượng biểu diễn chỉ để vẽ lại một khung trắng.

```bibtex
@article{sun2022drone,
  title   = {Drone-Based RGB-Infrared Cross-Modality Vehicle Detection via
             Uncertainty-Aware Learning},
  author  = {Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2022}
}
```

---

## 2. LLVIP — cảnh đêm, căn chỉnh chặt nhất

| | |
|---|---|
| Kaggle | [`monishshrivastava1/llvip-dataset`](https://www.kaggle.com/datasets/monishshrivastava1/llvip-dataset) |
| Gốc | https://bupt-ai-cz.github.io/LLVIP/ |
| Quy mô | ~15.488 cặp (30.976 ảnh) |
| Kích thước | 1280×1024 |
| Góc nhìn | Camera gắn cao nhìn chéo xuống đường phố, 26 vị trí |
| License | **Chỉ học thuật / phi thương mại** |

Chụp bằng rig binocular HIKVISION, đồng bộ cả thời gian lẫn không gian — nên đây
là nguồn có chất lượng ghép cặp tốt nhất trong ba nguồn dùng để train. Bổ sung
cảnh đêm và người đi bộ, hai thứ DroneVehicle tương đối ít.

```bibtex
@inproceedings{jia2021llvip,
  title     = {LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},
  author    = {Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and
               Zhou, Wenli},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV) Workshops},
  year      = {2021}
}
```

---

## 3. Teledyne FLIR ADAS Thermal Dataset v2 — bối cảnh mặt đất

| | |
|---|---|
| Kaggle | [`samdazel/teledyne-flir-adas-thermal-dataset-v2`](https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2) |
| Gốc | https://www.flir.com/oem/adas/adas-dataset-form/ |
| Quy mô | ~14.452 frame có annotation |
| Góc nhìn | Camera gắn trên xe, giao thông đô thị |
| License | Điều khoản riêng của Teledyne FLIR (nghiên cứu) |

> ### ⚠️ Cảnh báo căn chỉnh
> Camera RGB và camera nhiệt của FLIR có **field of view và độ phân giải khác
> nhau**. Các cặp ảnh đồng bộ về thời gian nhưng chỉ **căn chỉnh gần đúng** về
> không gian — khác hẳn DroneVehicle và LLVIP.
>
> `preprocess.rgb.fov_crop` trong `configs/datasets/flir_v2.yaml` là **ước lượng
> ban đầu, không phải kết quả hiệu chuẩn**. Trước khi tin dataset này:
>
> ```bash
> python scripts/check_alignment.py --dataset flir_v2 --n 6
> ```
>
> Nhìn panel thứ 4 (biên nhiệt màu đỏ chồng lên RGB). Nếu biên không nằm trên
> công trình/xe cộ, thử giá trị khác bằng `--fov-crop 0.55`, quét vài giá trị rồi
> ghi giá trị tốt nhất vào YAML. Nếu không cách nào khớp, đặt `enabled: false` —
> **ít dữ liệu vẫn tốt hơn dữ liệu dạy model làm mờ.**
>
> Đây cũng là lý do dataset này chỉ có `sample_weight: 0.10`.

Hai modality không dùng chung tên file (mỗi bên có hash riêng), nên
`adapters/flir_adas_v2.py` ghép cặp theo `(video_id, frame_number)` phân tích từ
tên file, thay vì so khớp chuỗi.

---

## 4. HIT-UAV — **chỉ dùng để đánh giá**

| | |
|---|---|
| Kaggle | [`pandrii000/hituav-a-highaltitude-infrared-thermal-dataset`](https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset) |
| Gốc | https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset |
| Quy mô | ~2.898 ảnh nhiệt |
| Góc nhìn | UAV độ cao 60–130 m |
| License | CC BY 4.0 |

**Dataset này không có ảnh RGB ghép cặp**, nên không thể dùng để train paired.
Mọi record của nó được gán `split=reference` và không bao giờ được sampler lấy.

Giá trị của nó nằm ở chỗ khác: nó là **phân phối thermal UAV thật** để tính FID
không cặp. Các chỉ số paired chỉ trả lời "có khớp đúng frame ground-truth này
không"; so sánh phân phối trả lời câu hỏi quan trọng hơn cho bài toán UAV —
"ảnh sinh ra có trông giống ảnh nhiệt UAV thật hay không".

```bibtex
@article{suo2023hituav,
  title   = {HIT-UAV: A high-altitude infrared thermal dataset for
             Unmanned Aerial Vehicle-based object detection},
  author  = {Suo, Jiashun and Wang, Tianyi and Zhang, Xingzhou and
             Chen, Haiyang and Zhou, Wei and Shi, Weisong},
  journal = {Scientific Data},
  volume  = {10},
  year    = {2023}
}
```

---

## Tại sao gộp bốn dataset này

Một model chỉ học từ DroneVehicle sẽ chỉ biết một loại cảnh, một camera nhiệt và
một khoảng độ cao. Mỗi nguồn bổ sung một trục khác nhau:

| Nguồn | Đóng góp | Weight |
|---|---|---|
| DroneVehicle | Đúng domain UAV; ngày và đêm | 0.60 |
| LLVIP | Ánh sáng yếu, người đi bộ, ghép cặp chuẩn nhất | 0.30 |
| FLIR v2 | Góc nhìn ngang tầm mắt, cảnh đô thị đa dạng | 0.10 |
| HIT-UAV | Độ cao lớn — dùng làm chuẩn đánh giá | — (0.0) |

Trọng số được khai báo trong `configs/pix2pix_uav.yaml` và được thực thi bằng
weighted sampler, **không** phải bằng cách nối dữ liệu. Nếu chỉ nối rồi shuffle,
tỉ lệ trộn sẽ do kích thước bản tải về quyết định thay vì do thiết kế.

---

## Nghĩa vụ khi sử dụng

- Trích dẫn dataset gốc trong mọi báo cáo, luận văn hay bài báo.
- Tôn trọng giới hạn phi thương mại (đặc biệt LLVIP).
- Không phân phối lại ảnh gốc; hãy dẫn link tới dataset trên Kaggle.
- Khi công bố ảnh nhiệt sinh ra, nói rõ đó là **ảnh tổng hợp, không radiometric**
  (xem phần Giới hạn trong [`README.md`](README.md)).
