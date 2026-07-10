# Syntopia — PRISMA-ADC

**PR**édiction par **I**A et **S**imulation des **M**écanismes d'**A**ction des **ADC**

Repo nghiên cứu cho Lot 1 của dự án PRISMA-ADC: xây dựng một simulateur biophysique (giải PDE khuếch tán–phản ứng) và pre-train một décodeur IA (surrogate neural network) học ánh xạ từ tham số vật lý sang các chỉ số không gian mô tả sự thấm/hiệu quả của kháng thể–thuốc trong mô u.

Trạng thái hiện tại: bài toán khởi động 1D (khuếch tán–phân rã, tuyến tính) đã được triển khai đầy đủ và kiểm chứng trong [src/diffusion_degradation/](src/diffusion_degradation/). Đây là bước rút gọn của Lot 1 đầy đủ (Krogh cylinder, đa biến, phi tuyến) mô tả trong dossier dự án.

---

## 1. Introduction

### 1.1 Bối cảnh

Antibody-Drug Conjugate (ADC) là một lớp thuốc ung thư kết hợp kháng thể nhắm đích, một linker, và một payload gây độc tế bào. Không gian thiết kế ADC rất lớn (~10⁵ tổ hợp dose × payload × linker × DAR × site conjugaison), nhưng một chương trình tiền lâm sàng cổ điển chỉ kiểm tra được 20–30 cấu trúc *in vitro* và <5 *in vivo* trong 6–12 tháng, chi phí >500 k€, tỷ lệ thành công lâm sàng <10%. Vấn đề không phải thiếu phân tử để thử — mà là không đủ nhanh/rẻ để tìm ra phân tử tốt.

Nút thắt cốt lõi: hiệu quả ADC phụ thuộc vào hành vi *không gian* của nó trong mô u — khuếch tán từ mạch máu, bão hoà thụ thể ở rìa u ("binding site barrier") chặn đường vào sâu, nội hoá tế bào, và lan payload sang tế bào lân cận ("bystander effect"). Đây là quá trình hiện **không thể quan sát trực tiếp**: nuôi cấy 2D không có tổ chức không gian, còn mô hình động vật (PDX) thì chậm, tốn kém, và kém đại diện cho người.

### 1.2 Chiến lược PRISMA-ADC

Dự án kết hợp hai công cụ:

1. **Mô hình biophysique** (Krogh cylinder — Thurber & Wittrup 2007/2012, mở rộng bởi Cilliers et al. 2016 cho ADC) — đúng về cơ chế nhưng cứng: đòi tham số biết trước, chỉ dự đoán được cho tổ hợp đã biết, hình học 1D, môi trường đồng nhất.
2. **Mô hình IA** — học end-to-end từ mô tả phân tử đến hiệu quả không gian, tổng quát hoá được sang cấu trúc mới, nhưng cần dữ liệu.

Cách phối hợp: dùng (1) để sinh dữ liệu tổng hợp số lượng lớn, pre-train (2) trên dữ liệu đó, sau đó tinh chỉnh (2) trên dữ liệu thực nghiệm *in vitro*, và cuối cùng dùng active learning để chọn thí nghiệm tiếp theo thông tin nhất. Bốn Lot của Phase 1: Lot 1 (simulateur + décodeur, dữ liệu tổng hợp), Lot 2 (dữ liệu *in vitro*, encodeur phân tử), Lot 3 (apprentissage actif), Lot 4 (coordination/IP).

### 1.3 Phạm vi repo này

Repo hiện triển khai **bài toán khởi động của Lot 1**: phương trình khuếch tán–phân rã 1D tuyến tính, dùng đúng hình học và điều kiện biên của Krogh cylinder, nhưng bỏ qua các phi tuyến (bão hoà thụ thể) và các biến ADC bổ sung (giải phóng payload, bystander). Mục tiêu: kiểm chứng toàn bộ pipeline "solver → dữ liệu tổng hợp → surrogate NN → đánh giá" trên bài toán đơn giản nhất có thể, trước khi mở rộng sang mô hình ADC đầy đủ.

---

## 2. Background knowledge

Kiến thức nền cần có để hiểu và mở rộng repo này, chia theo 3 mảng. Chi tiết đầy đủ hơn (lộ trình học, thời lượng, bài tập tự kiểm tra) xem [docs/LO_TRINH_HOC.md](docs/LO_TRINH_HOC.md).

### A. Sinh học ADC & vận chuyển thuốc trong u (mức khái niệm)
- Kháng thể (IgG, ~150 kDa) khuếch tán chậm trong mô; ADC = kháng thể + linker + payload, 5 tham số thiết kế (dose, payload, linker, DAR, site conjugaison).
- Antigen/thụ thể, nội hoá (internalization), giải phóng payload.
- 4 hiện tượng không gian: diffusion, binding site barrier, internalisation, bystander effect.
- Vì sao mô hình 2D/PDX không đủ để dự đoán hành vi trong mô người.

### B. Toán & vật lý mô phỏng (mức làm được)
- PDE khuếch tán–phản ứng `∂c/∂t = D ∂²c/∂x² − r·c`; độ dài thấm `λ = √(D/r)`.
- Điều kiện biên Dirichlet (nguồn cố định tại thành mạch) và Neumann/zero-flux (đối xứng ở biên ngoài mô).
- Nghiệm giải tích: steady-state + khai triển theo hàm riêng (eigenfunction expansion) cho nghiệm chuyển tiếp.
- Giải số: sai phân hữu hạn (FDM), sơ đồ Crank–Nicolson (ẩn, ổn định vô điều kiện), thuật toán Thomas cho hệ ba đường chéo.
- Mô hình Krogh cylinder và số Biot `Bi = 2PR_cap/(Dε)` — vì sao cho phép rút bài toán 3D về mô hình compartmental.

### C. Machine Learning (mức làm được)
- Nền tảng: train/val/test split, overfitting, gradient descent.
- MLP: fully-connected, ReLU/GELU, kiến trúc surrogate 4 → (256,256,256) → N_grid.
- Surrogate modeling: học ánh xạ `tham số vật lý p → profile y` để thay thế solver chậm.
- Kỹ thuật huấn luyện: chuẩn hoá input, log-transform output `log(C+ε)`, chỉ số MAE/RMSE/R².
- (Cho Lot 2, học sau) Biểu diễn phân tử: SMILES, ECFP, RDKit/Mordred, encodeur `x → p`.
- (Để biết) PINN/Universal Differential Equations — vì sao dossier chọn surrogate offline thay vì hybrid tích hợp.
- (Cho Lot 3, học sau) Active learning: acquisition function, Monte Carlo Dropout cho uncertainty.

---

## 3. Literature review

| Nguồn | Đóng góp cho repo này |
|---|---|
| **Thurber & Wittrup (2012)**, *A mechanistic compartmental model for total antibody uptake in tumors*, J Theor Biol 314:57-68 (`papers/2012 Thurber cylinder JTB.pdf`) | Nguồn hình học và điều kiện biên: mô hình Krogh cylinder (Fig. 1b), phương trình khuếch tán–liên kết–nội hoá, dải giá trị tham số thực nghiệm (Table 1: D, k_e, K_d, [Ag]...) dùng làm cơ sở cho khoảng sampling trong `generate_data.py`. Khái niệm số Biot giải thích vì sao mô hình compartmental 0-D là xấp xỉ hợp lệ khi extravasation là bước giới hạn tốc độ. |
| **Cilliers et al. (2016)**, *Multiscale Modeling of Antibody-Drug Conjugates*, AAPS J 18(5):1117-1130 | Mở rộng mô hình Krogh sang ADC đầy đủ: thêm nội hoá, giải phóng payload, bystander effect. Là đích đến của Tâche 1.4/Lot 2 — bài 1D hiện tại là bản rút gọn tuyến tính của mô hình này. |
| **Raissi, Perdikaris, Karniadakis (2019)**, *Physics-informed neural networks*, J Comput Phys 378:686-707 | Hướng "hybrid tích hợp" (PINN) mà dossier PRISMA-ADC **cân nhắc và không chọn** cho Lot 1, vì lý do ổn định tính toán và tính linh hoạt kiến trúc — chọn surrogate offline (train trên dữ liệu tổng hợp có sẵn) thay vì ràng buộc PDE trực tiếp vào loss. |
| **Reker & Schneider (2015)**, *Active-learning strategies in computer-assisted drug discovery*, Drug Discov Today 20(4):458-465 | Cơ sở lý thuyết cho Lot 3 (apprentissage actif) — chọn thí nghiệm thông tin nhất thay vì ngẫu nhiên, đã có tiền lệ giảm 10× số thí nghiệm trong không gian tổ hợp thuốc (Wang et al. 2025, nội bộ nhóm). |
| **Đặc tả kỹ thuật nội bộ** (`papers/NN_surrogate_diffusion_degradation.pdf`) | Bản đặc tả trực tiếp của bài toán khởi động 1D triển khai trong repo này — input/output, kiến trúc MLP đề xuất, chỉ số đánh giá. |
| **Dossier Pionniers de l'IA** (`papers/Dossier Pionniers de l'IA v26.05.31_PN2_AR_share.pdf`) | Bối cảnh dự án đầy đủ: état de l'art, 4 Lot của Phase 1, quản lý rủi ro, sở hữu trí tuệ. Nguồn cho mục 1 và bảng Lot ở trên. |

Ghi chú khoảng trống: chưa có review riêng về ADC pharmacology tổng quát (DAR, cơ chế bystander ở mức phân tử) hay về biểu diễn phân tử (SMILES/ECFP) — các mảng này cần cho Lot 2, xem mục tham khảo mở rộng trong [docs/LO_TRINH_HOC.md](docs/LO_TRINH_HOC.md) mục 4.

---

## 4. Methods

### 4.1 Mô hình vật lý

```
∂c/∂t = D ∂²c/∂x² − r·c ,      x ∈ [0, L]

Biên:   c(0, t)     = c₀     (Dirichlet — nguồn cố định, thành mạch)
        ∂c/∂x(L, t) = 0      (Neumann/zero-flux — đối xứng biên ngoài mô)
Đầu:    c(x, 0)     = 0
```

`D` = hệ số khuếch tán (µm²/s), `r` = tốc độ mất phân tử bậc 1 (nội hoá/phân rã, s⁻¹), `L` = bán kính Krogh cylinder (µm).

### 4.2 Nghiệm giải tích ([solver.py](src/diffusion_degradation/solver.py))

- **Steady-state** (`steady_state_profile`): `c_ss(x) = c₀ · cosh((L−x)/λ) / cosh(L/λ)`, với `λ = √(D/r)` là độ dài thấm.
- **Transient** (`analytical_transient_profile`): khai triển `c(x,t) = c_ss(x) − w(x,t)`, với `w` giải cùng PDE nhưng biên thuần nhất; hàm riêng `sin(k_n x)`, `k_n = (2n−1)π/(2L)`, suy giảm `exp(−(D k_n² + r)t)`. Hệ số `B_n` tính bằng phép chiếu lên cơ sở trực giao (tích phân trapezoid, 200 mode mặc định).

### 4.3 Nghiệm số (`fdm_crank_nicolson`)

Sai phân hữu hạn ẩn Crank–Nicolson (ổn định vô điều kiện), giải hệ ba đường chéo bằng thuật toán Thomas ở mỗi bước thời gian. Biên Neumann tại `x=L` cài đặt bằng nút ma "gương" (`c[nx] = c[nx-2]`). Mỗi thời điểm `t` yêu cầu được march độc lập từ `t=0` với số bước riêng (heuristic theo độ chính xác, giới hạn `max_steps`) — tránh dưới-phân giải các `t` nhỏ khi `t_max` lớn hơn nhiều bậc độ lớn. Dùng để (a) kiểm chứng chéo với nghiệm giải tích, (b) làm nền cho Tâche 1.4 khi `D(x), r(x)` không còn đồng nhất.

### 4.4 Sinh dữ liệu tổng hợp ([generate_data.py](src/diffusion_degradation/generate_data.py))

- Sample `(c₀, D, r)` log-uniform trong khoảng lấy cảm hứng từ Table 1 [Thurber 2012]: `c₀ ∈ [10⁻², 10¹]`, `D ∈ [1, 50] µm²/s`, `r ∈ [10⁻⁶, 10⁻³] s⁻¹`.
- Với mỗi điều kiện, tính profile tại `n_times` thời điểm log-spaced trên `[T_max/1000, T_max]` (`T_max` = 3 ngày) bằng nghiệm giải tích (nhanh, chính xác cho PDE tuyến tính đồng nhất).
- Cross-check một tập con ngẫu nhiên (≤20 điều kiện) với `fdm_crank_nicolson` độc lập, in sai số tương đối worst-case, để đảm bảo công thức giải tích không bị dùng sai trước khi gán nhãn hàng loạt.
- Input mô hình: `X = (log₁₀c₀, log₁₀D, log₁₀r, t)`; output: `Y = C(x,t)` trên lưới `x` cố định (100 điểm).

### 4.5 Kiến trúc surrogate ([model.py](src/diffusion_degradation/model.py))

MLP: input 4 → 3 lớp ẩn (mặc định 256-256-256, activation GELU) → output N_grid (100). Đây chính là "décodeur biophysique" `p → y` trong dossier.

### 4.6 Huấn luyện ([train.py](src/diffusion_degradation/train.py))

- Split 70/15/15 (train/val/test), ngẫu nhiên theo hàng (mỗi hàng = một cặp (tham số, t) độc lập).
- Chuẩn hoá input theo (mean, std) của tập train.
- Output train trong log-space: `log(C + ε)` rồi chuẩn hoá — cần thiết vì nồng độ trải nhiều bậc độ lớn.
- Loss MSE, optimizer Adam, early stopping theo validation loss (patience mặc định 20 epoch).
- Đánh giá trên test set bằng MAE/RMSE/R², cả trong log-space lẫn sau khi biến đổi ngược về đơn vị vật lý.

### 4.7 Đánh giá ([evaluate.py](src/diffusion_degradation/evaluate.py))

Nạp lại đúng test split đã lưu trong checkpoint, tính lại metrics, vẽ (a) profile true-vs-predicted cho vài điều kiện mẫu, (b) scatter plot true-vs-predicted trên toàn test set kèm R².

### 4.8 Kết quả kiểm chứng thực nghiệm (đã chạy thật)

Analytical vs FDM (`D=10 µm²/s, r=10⁻³ s⁻¹, L=100 µm`): sai số tương đối tối đa từ 2.7×10⁻⁵ (t=50s) xuống 2.1×10⁻⁶ (t≥5000s) — hai phương pháp độc lập khớp nhau tới ~10⁻⁵. Sinh dữ liệu: ~5.4 ms/điều kiện ⟹ 100 000 điều kiện × 8 thời điểm ước tính ~9–10 phút trên CPU. Chi tiết đầy đủ và các lưu ý kỹ thuật đã gặp (time-stepping, `np.trapezoid` vs `np.trapz`...) xem [docs/HUONG_DAN_PROJECT.md §4.8](docs/HUONG_DAN_PROJECT.md).

---

## 5. Tutorial

### 5.1 Cài đặt

```bash
cd /home/duc-khiem/Documents/SummerInternship/SYNTOPIA
pip install -r requirements.txt   # numpy, torch, matplotlib
```

### 5.2 Chạy end-to-end

Tất cả lệnh chạy từ `src/diffusion_degradation/`:

```bash
cd src/diffusion_degradation

# 1. Tự kiểm tra solver: so nghiệm giải tích với FDM, in sai số tương đối
python3 solver.py

# 2. Sinh dữ liệu tổng hợp (demo nhanh — outputs/dataset.npz đã có sẵn, 2000 điều kiện x 8 thời điểm)
python3 generate_data.py --n_conditions 2000 --n_times 8 --out ../../outputs/dataset.npz

# Cho deliverable Lot 1 đầy đủ (Tâche 1.2 đòi 100 000 điều kiện):
python3 generate_data.py --n_conditions 100000 --n_times 8 --out ../../outputs/dataset_full.npz

# 3. Train surrogate
python3 train.py --dataset ../../outputs/dataset.npz --epochs 200

# 4. Đánh giá + vẽ hình so sánh true vs predicted
python3 evaluate.py --dataset ../../outputs/dataset.npz
```

Kết quả: `outputs/surrogate_model.pt` (checkpoint), `outputs/training_history.json`, `outputs/test_metrics.json`, `outputs/figures/profiles_true_vs_pred.png`, `outputs/figures/scatter_true_vs_pred.png`.

### 5.3 Thí nghiệm gợi ý (để học, không bắt buộc)

- Đổi `D`, `r` trong `solver.py __main__` → quan sát `λ=√(D/r)` thay đổi hình dạng profile steady-state.
- Đổi `--hidden_dims` (ví dụ `128,128` hoặc `256,256,256,256`) và `--activation relu` khi gọi `train.py` → so sánh RMSE/R² trên test set.
- Bỏ log-transform trong `train.py` (train trực tiếp trên `Y` thay vì `log(Y+EPS)`) → quan sát metrics tệ đi, để hiểu tại sao log-transform cần thiết khi nồng độ trải nhiều bậc độ lớn.
- Tăng `max_steps` trong lệnh gọi `fdm_crank_nicolson` ở `build_dataset()` nếu cần validation chặt hơn 10⁻³.

### 5.4 Đường dẫn học tiếp

- Lý thuyết chi tiết từng dòng công thức + code đầy đủ inline: [docs/HUONG_DAN_PROJECT.md](docs/HUONG_DAN_PROJECT.md).
- Lộ trình học nền tảng theo giai đoạn (0→3), tham khảo phân loại theo chủ đề, 8 câu tự kiểm tra: [docs/LO_TRINH_HOC.md](docs/LO_TRINH_HOC.md).
- Hướng mở rộng sang Lot 1 đầy đủ (heterogeneity, `D(x)`/`r(x)`) và Lot 2 (encodeur phân tử, kiến trúc end-to-end): [docs/HUONG_DAN_PROJECT.md §5](docs/HUONG_DAN_PROJECT.md).
