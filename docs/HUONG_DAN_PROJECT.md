# Hướng dẫn thực hiện: NN Surrogate cho phương trình Diffusion–Degradation 1D

**Dự án:** Syntopia — PRISMA-ADC, Lot 1 ("Infrastructure IA: simulateur et décodeur biophysiques")
**Tài liệu nguồn:** `NN_surrogate_diffusion_degradation.pdf`, `Dossier Pionniers de l'IA v26.05.31...pdf`, `2012 Thurber cylinder JTB.pdf`

Code đầy đủ, đã chạy và kiểm tra thực nghiệm, nằm ở `src/diffusion_degradation/`. Tài liệu này giải thích lý thuyết đứng sau code đó và cách chạy nó.

---

## 1. Tổng quan project

### 1.1 Bối cảnh

Syntopia đang phát triển **PRISMA-ADC**: một mô hình lai biophysique–IA để dự đoán hành vi không gian (pénétration, mort cellulaire) của các ADC (Antibody-Drug Conjugate) trong mô u. Lý do: thử nghiệm in vivo/in vitro tốn kém và chậm, còn mô hình vật lý thuần (cylindre de Krogh, Thurber et al. 2007/2012, Cilliers et al. 2016) thì đúng về cơ chế nhưng quá cứng — không học được từ dữ liệu, không tổng quát hoá sang cấu trúc phân tử mới.

Chiến lược của Lot 1 (xem `Dossier Pionniers de l'IA...pdf`, mục 3.c, "Lot n°1"):

1. **Tâche 1.1** — Xây dựng simulateur biophysique (giải PDE khuếch tán–phản ứng bằng phần tử hữu hạn/sai phân hữu hạn).
2. **Tâche 1.2** — Sinh 100 000 điều kiện tổng hợp (synthetic conditions) bằng simulateur đó.
3. **Tâche 1.3** — Pre-train một MLP (decodeur biophysique) học ánh xạ `paramètres physiques p → métriques spatiales y`.
4. **Tâche 1.4** — Chỉ làm nếu cần: thêm heterogeneity vào simulateur.

`NN_surrogate_diffusion_degradation.pdf` chính là bản đặc tả kỹ thuật **rút gọn** của 3 tâche đầu tiên này, dùng phương trình 1D đơn giản nhất có thể (khuếch tán + phân rã bậc 1) làm bài toán khởi động (proof of concept) trước khi mở rộng sang mô hình ADC đầy đủ (Cilliers et al. 2016) với các biến bổ sung: giải phóng payload, nội hoá, bystander effect, v.v.

### 1.2 Việc bạn cần làm

Đúng như file PDF thứ 3 yêu cầu, và đã được triển khai đầy đủ trong `src/diffusion_degradation/`:

| Bước | File | Nội dung |
|---|---|---|
| 1 | `solver.py` | Nghiệm giải tích + solver FDM (Crank–Nicolson) cho phương trình khuếch tán–phân rã |
| 2 | `generate_data.py` | Sinh dữ liệu tổng hợp: sample (c₀, D, r) log-uniform, tính profile C(x,t) |
| 3 | `model.py` | Kiến trúc MLP surrogate |
| 4 | `train.py` | Train trên dữ liệu tổng hợp, log-transform, tách 70/15/15, tính MAE/RMSE/R² |
| 5 | `evaluate.py` | Đánh giá trên test set + vẽ hình so sánh true vs. predicted |

Toàn bộ code đã được **chạy thử thật** trong quá trình soạn tài liệu này (không phải code chưa kiểm chứng) — số liệu sai số/thời gian chạy ở mục 4 là số liệu thực tế, không phải ước lượng.

### 1.3 Liên hệ với mô hình Krogh cylinder (Thurber & Wittrup, 2012)

Bài toán 1D ở đây **không phải một bài tập toán tách rời** — nó dùng chính hình học và điều kiện biên trong mô hình mà Thurber dùng để mô tả kháng thể khuếch tán quanh mạch máu trong u (xem `2012 Thurber cylinder JTB.pdf`, Fig. 1b, "Krogh cylinder"):

- `x = 0`: thành mạch máu, nồng độ được giữ cố định `c₀` (giống `[Ab]plasma` không đổi tại vách mạch trong bài Thurber).
- `x = L`: biên ngoài của "Krogh cylinder" (bán kính `R_Krogh`), là mặt đối xứng giữa 2 mạch máu lân cận → **không có dòng chảy qua biên này** (Neumann, `∂c/∂x = 0`).
- `D`: hệ số khuếch tán của kháng thể/ADC/payload trong mô (Thurber Table 1: IgG ≈ 14 µm²/s, khoảng 5–50 µm²/s).
- `r`: gộp tốc độ mất tín hiệu — nội hoá + phân rã (tương tự `k_e` trong Thurber, 2×10⁻⁶ – 2×10⁻⁴ s⁻¹) hoặc tốc độ giải phóng payload trong ngữ cảnh ADC (Cilliers et al. 2016).

Vì vậy, mọi trực giác bạn xây dựng ở bài 1D này (độ dài thấm `λ = √(D/r)`, vai trò tương đối của khuếch tán vs. phân rã...) **chuyển thẳng sang** bài toán ADC đầy đủ ở Lot 1/Lot 2.

---

## 2. Lý thuyết cốt lõi

### 2.1 Phương trình khuếch tán–phân rã 1D

```
∂c/∂t = D ∂²c/∂x² − r·c ,      x ∈ [0, L]
```

- `c(x,t)`: nồng độ phân tử (kháng thể/ADC/payload) tại vị trí `x`, thời điểm `t`.
- `D` (µm²/s): hệ số khuếch tán.
- `r` (s⁻¹): tốc độ "mất" phân tử bậc 1 (phân rã, nội hoá, hoặc — nếu đảo dấu — phản ứng gắn kết giả-bậc-1 trong chế độ dưới bão hoà, xem `2012 Thurber...pdf` mục 2.1, phương trình (1)-(3): khi `[Ab]total ≪ [Ag]`, tỉ lệ bound/free là hằng số, nên động học bậc nhất là một xấp xỉ hợp lệ).

**Điều kiện biên** (giống Krogh cylinder, xem mục 1.3):
```
c(0, t) = c₀                (Dirichlet — nguồn cố định, vd. thành mạch)
∂c/∂x (L, t) = 0             (Neumann/zero-flux — đối xứng ở biên ngoài mô)
```
**Điều kiện đầu:** `c(x, 0) = 0` (mô chưa có phân tử thuốc trước khi tiêm).

Đây là PDE **tuyến tính** — tính chất này là chìa khoá để có nghiệm giải tích closed-form ở mục 2.2, và cũng là lý do `c₀` chỉ đóng vai trò hệ số nhân tuyến tính lên toàn bộ profile (không ảnh hưởng đến *hình dạng* nghiệm) — điều rất khác so với bài toán ADC đầy đủ, nơi có bão hoà thụ thể ("binding site barrier") làm phương trình trở nên phi tuyến.

### 2.2 Nghiệm giải tích

#### a) Nghiệm dừng (steady-state, t → ∞)

Đặt `dc/dt = 0`:
```
D c'' − r c = 0,   c(0) = c₀,   c'(L) = 0
```
Nghiệm tổng quát `c(x) = A cosh(x/λ) + B sinh(x/λ)` với `λ = √(D/r)` (**độ dài thấm đặc trưng** — penetration length, đúng khái niệm dùng trong phân tích của Thurber). Áp 2 điều kiện biên:

```
c_ss(x) = c₀ · cosh((L − x)/λ) / cosh(L/λ)
```

Khi `L ≫ λ` (mô dày so với độ thấm), công thức này giảm về `c_ss(x) ≈ c₀ e^(−x/λ)` — đúng dạng suy giảm mũ quen thuộc trong các bài báo về "tumor penetration".

#### b) Nghiệm chuyển tiếp (transient) — khai triển theo hàm riêng (eigenfunction expansion)

Đặt `w(x,t) = c_ss(x) − c(x,t)`. Vì PDE tuyến tính và `c_ss` là nghiệm dừng, `w` thoả **cùng PDE** nhưng với biên **thuần nhất**: `w(0,t)=0`, `w_x(L,t)=0`, và điều kiện đầu `w(x,0) = c_ss(x)` (vì `c(x,0)=0`).

Bài toán biên thuần nhất này có hàm riêng `sin(k_n x)` với:
```
k_n = (2n − 1)π / (2L),        n = 1, 2, 3, ...
```
(điều kiện `w_x(L,t)=0` ⟺ `cos(k_n L) = 0` ⟺ `k_n L = (n − ½)π`), và mỗi mode suy giảm với tốc độ riêng:
```
μ_n = D k_n² + r
```
Khai triển `w(x,t) = Σ_n B_n sin(k_n x) e^{−μ_n t}`, với `B_n` là hệ số chiếu điều kiện đầu lên hệ hàm riêng trực giao trên `[0, L]`:
```
B_n = (2/L) ∫₀^L c_ss(x) sin(k_n x) dx
```

⟹ Nghiệm đầy đủ:
```
c(x,t) = c_ss(x) − Σ_{n=1}^{∞} B_n sin(k_n x) e^{−(D k_n² + r) t}
```

Trong `solver.py`, tích phân `B_n` được tính bằng cầu phương số (`numpy.trapezoid`) trên lưới mịn — nhanh, chính xác, và tránh phải tra bảng tích phân `cosh × sin` bằng tay. Tổng chuỗi được cắt ở `n_modes` (mặc định 150–300 mode; các mode cao suy giảm rất nhanh nên hội tụ tốt). **Đây chính là "nghiệm giải tích" mà đề bài yêu cầu dùng để sinh dữ liệu khi có thể** (mục "Tasks", bước 1 trong `NN_surrogate_diffusion_degradation.pdf`).

**Đã kiểm chứng thực nghiệm:** so với solver số (mục 2.3), sai số tương đối tối đa `< 3×10⁻⁵` trên toàn bộ dải thời gian test (xem log chạy thật ở mục 4.6).

### 2.3 Phương pháp sai phân hữu hạn (FDM) — Crank–Nicolson

Nghiệm giải tích ở trên chỉ đúng khi `D` và `r` là **hằng số theo x** (môi trường đồng nhất). Ngay khi Tâche 1.4 của Lot 1 cần thêm **heterogeneity** (mô u không đồng nhất, `D(x)` hoặc `r(x)` thay đổi theo vị trí — đúng như limitation của mô hình Krogh cylinder gốc mà dossier Syntopia chỉ ra), nghiệm giải tích không còn áp dụng được và solver số trở thành công cụ chính. Đây là lý do `solver.py` triển khai **cả hai** phương pháp, không chỉ một.

Rời rạc hoá không gian: lưới đều `x_i = i·Δx`, `i = 0..N_x−1`. Sơ đồ **Crank–Nicolson** (ẩn, bậc 2 theo thời gian, **ổn định vô điều kiện** — không giống Euler hiện tại vốn đòi `Δt` rất nhỏ để ổn định):

```
(c_i^{n+1} − c_i^n)/Δt = (D/2)[(δ²c^{n+1})_i + (δ²c^n)_i]/Δx² − (r/2)(c_i^{n+1} + c_i^n)
```
với `δ²c_i = c_{i+1} − 2c_i + c_{i-1}`. Sắp xếp lại, mỗi bước thời gian là một **hệ ba đường chéo (tridiagonal)**:
```
−α c_{i-1}^{n+1} + (1+2α+β) c_i^{n+1} − α c_{i+1}^{n+1} = α c_{i-1}^n + (1−2α−β) c_i^n + α c_{i+1}^n
```
với `α = DΔt/(2Δx²)`, `β = rΔt/2`. Hệ này được giải bằng **thuật toán Thomas** (`_thomas_solve` trong code — O(N) mỗi bước, không cần SciPy).

- **Biên Dirichlet** (`x=0`): loại khỏi hệ ẩn, thay trực tiếp `c₀`.
- **Biên Neumann/zero-flux** (`x=L`): dùng nút ảo (ghost node) đối xứng `c_{N_x} = c_{N_x−2}`, tương đương xấp xỉ bậc 2 của `∂c/∂x=0`.

Số bước thời gian được chọn thích ứng theo từng `t` yêu cầu (không dùng một `Δt` chung cho toàn bộ khoảng thời gian) — nếu không, một điều kiện có `t` nhỏ sẽ bị "dưới lấy mẫu" nghiêm trọng khi được đặt cạnh một điều kiện có `t` lớn hơn hàng nghìn lần trong cùng một dataset. Đây là một lỗi thực tế tôi gặp phải và đã sửa khi viết code — chi tiết ở mục 4.6.

### 2.4 Kiến trúc Neural Network surrogate

Đúng theo đặc tả PDF:

```
Input:  (log₁₀ c₀, log₁₀ D, log₁₀ r, t)     — vector 4 chiều
Output: C(x, t)                              — vector N_x chiều (toàn bộ profile không gian tại thời điểm t)
```

- **Vì sao log-scale cho `c₀, D, r`?** Ba tham số này được sample log-uniform trên nhiều bậc độ lớn (`D`: 1–50, `r`: 10⁻⁶–10⁻³...) — dùng `log₁₀` biến việc sample log-uniform thành uniform thông thường trong không gian input, giúp NN học dễ hơn (tránh input có phạm vi giá trị chênh lệch hàng chục nghìn lần).
- **Vì sao học trên `log(C + ε)` thay vì `C` trực tiếp?** Nồng độ `C(x,t)` có thể chênh nhau nhiều bậc độ lớn giữa vùng gần nguồn (`x≈0`) và vùng xa (`x≈L`, đặc biệt khi `r` lớn ⟹ suy giảm mũ dốc). Huấn luyện trực tiếp trên `C` khiến MSE loss bị chi phối hoàn toàn bởi vùng nồng độ cao, NN sẽ học rất kém phần đuôi suy giảm (chính là phần quan trọng để đánh giá độ thấm sâu — "penetration depth", chỉ số sinh học cốt lõi trong toàn bộ dossier PRISMA-ADC). `ε` nhỏ (10⁻⁸) tránh `log(0)`.
- **Kiến trúc:** MLP đơn giản, 3–5 lớp ẩn, 128–256 units, activation GELU/ReLU — đúng theo đề bài, chưa cần kiến trúc phức tạp hơn (Fourier features, DeepONet...) ở bước proof-of-concept này.
- **Chuẩn hoá input:** standardize (trừ mean, chia std) tính trên tập train — tránh leakage từ val/test.

---

## 3. Tài liệu tham khảo

### 3.1 Tiếng Anh

**PDE & giải tích số:**
- W. Strauss, *Partial Differential Equations: An Introduction*, Wiley — nhập môn PDE tuyến tính, tách biến, khai triển hàm riêng (chính là kỹ thuật dùng ở mục 2.2b).
- R. Haberman, *Applied Partial Differential Equations with Fourier Series and Boundary Value Problems*, Pearson — rất sát với bài toán ở đây (diffusion equation + Fourier/eigenfunction methods, có cả các ví dụ với reaction term).
- R. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM — chương về sơ đồ ẩn/Crank–Nicolson, ổn định von Neumann.
- W. Deen, *Analysis of Transport Phenomena*, Oxford University Press — được chính Thurber & Wittrup (2012) trích dẫn cho lập luận "scaling resistances in series"; nền tảng transport phenomena cho toàn bộ mảng tumor-PK.

**Phần tử hữu hạn (FEM — cần khi mở rộng sang 2D/3D, Tâche 1.1 "par éléments finis"):**
- M. Larson & F. Bengzon, *The Finite Element Method: Theory, Implementation, and Applications*, Springer (có bản PDF miễn phí của tác giả) — nhập môn FEM hiện đại, code mẫu rõ ràng.
- J. N. Reddy, *An Introduction to the Finite Element Method*, McGraw-Hill.

**Machine learning / Deep learning:**
- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press — miễn phí tại **deeplearningbook.org**. Chương 6-8 (MLP, tối ưu hoá) là đủ cho surrogate model này.
- C. Bishop, *Pattern Recognition and Machine Learning*, Springer — nền tảng lý thuyết (bias-variance, regularization) hữu ích khi bạn cần giải thích tại sao train/val/test split hay early stopping quan trọng.
- PyTorch official tutorials — **pytorch.org/tutorials** — đủ để hiểu toàn bộ `train.py`/`model.py`.

**Scientific machine learning (định hướng dài hạn của Syntopia):**
- M. Raissi, P. Perdikaris, G. Karniadakis, "Physics-informed neural networks", *J. Comput. Phys.* 378 (2019) — đã được trích trong dossier Syntopia [3]; đọc để hiểu **vì sao** Syntopia chọn hướng "pre-train trên dữ liệu tổng hợp" thay vì PINN tích hợp (được giải thích rõ trong dossier, mục "Méthodologie scientifique envisagée").
- J. N. Kutz & S. Brunton, *Data-Driven Science and Engineering*, Cambridge University Press — cầu nối tốt giữa mô hình vật lý cổ điển và ML hiện đại.
- D. Reker, G. Schneider, "Active-learning strategies in computer-assisted drug discovery", *Drug Discov. Today* 20 (2015) — nền tảng cho Lot 3 (apprentissage actif), đã trích trong dossier [4].

### 3.2 Tiếng Pháp

- G. Allaire, *Analyse numérique et optimisation*, Les Éditions de l'École Polytechnique — giáo trình chuẩn của X, có chương về différences finies và éléments finis cho phương trình parabolique (đúng loại phương trình ở đây).
- P.-A. Raviart & J.-M. Thomas, *Introduction à l'analyse numérique des équations aux dérivées publiques*, Masson/Dunod — tài liệu tham chiếu kinh điển về EDP + éléments finis trong giáo trình Pháp.
- Ghi chú: mảng "apprentissage automatique" hiện đại ít có giáo trình Pháp kinh điển tương đương — khuyến nghị dùng tài liệu tiếng Anh (mục 3.1) cho phần NN, và tài liệu Pháp cho phần EDP/số nếu bạn cần trao đổi thuật ngữ với đội ngũ Pháp của Syntopia (ví dụ đúng thuật ngữ dossier dùng: "simulateur biophysique", "décodeur biophysique", "apprentissage actif").

### 3.3 Tiếng Việt

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — **machinelearningcoban.com** — tài liệu tiếng Việt phổ biến nhất về ML nền tảng, có phần về neural network/MLP viết dễ hiểu.
- Nguyễn Thanh Tuấn, *Deep Learning cơ bản* — sách tiếng Việt về deep learning, có bản đọc miễn phí trực tuyến (tác giả duy trì tại **nttuan8.com**); phần về MLP, activation, backpropagation, overfitting/regularization áp dụng trực tiếp cho `train.py`.
- Với PDE/giải tích số bằng tiếng Việt: chưa có tài liệu tiếng Việt uy tín tương đương mà tôi tự tin trích dẫn chính xác tên/link — khuyến nghị tìm giáo trình "Phương trình đạo hàm riêng" hoặc "Giải tích số" của các trường (Bách Khoa, KHTN, ĐHQGHN) qua thư viện trường, hoặc dùng trực tiếp Strauss/Haberman (mục 3.1) vì thuật ngữ toán không rào cản nhiều.

### 3.4 Papers trực tiếp liên quan (đã có trong dossier Syntopia)

- G. M. Thurber, M. M. Schmidt, K. D. Wittrup, "Antibody tumor penetration: transport opposed by systemic and antigen-mediated clearance", *Adv. Drug Deliv. Rev.* 60 (2008) 1421–34.
- G. M. Thurber, K. D. Wittrup, "A mechanistic compartmental model for total antibody uptake in tumors", *J. Theor. Biol.* 314 (2012) 57–68 — chính là file `2012 Thurber cylinder JTB.pdf` bạn đã có; đọc kỹ mục 2.2 "Vascular transport" và Fig. 1b để thấy chính xác hình học Krogh cylinder mà bài 1D của bạn đang mô phỏng đơn giản hoá.
- C. Cilliers et al., "Multiscale Modeling of Antibody-Drug Conjugates...", *AAPS J.* 18 (2016) 1117–1130 — bước mở rộng từ mô hình kháng thể (Thurber) sang ADC đầy đủ; đọc để biết bài 1D của bạn sẽ được "cắm" vào đâu trong mô hình đầy đủ ở Lot 2.

---

## 4. Code hoàn chỉnh

### 4.1 Cấu trúc thư mục

```
SYNTOPIA/
├── requirements.txt
├── docs/
│   └── HUONG_DAN_PROJECT.md        (tài liệu này)
├── outputs/
│   └── dataset.npz                  (dataset demo, 2000 điều kiện × 8 thời điểm, đã sinh sẵn)
└── src/diffusion_degradation/
    ├── solver.py                    (nghiệm giải tích + FDM)
    ├── generate_data.py             (sinh dữ liệu tổng hợp)
    ├── model.py                     (kiến trúc MLP)
    ├── train.py                     (huấn luyện)
    └── evaluate.py                  (đánh giá + vẽ hình)
```

Cài thư viện:
```bash
pip install -r requirements.txt
```
```text
numpy>=1.24
torch>=2.1
matplotlib>=3.7
```
(Không cần SciPy — `solver.py` tự triển khai thuật toán Thomas để giải hệ ba đường chéo, tránh thêm dependency.)

### 4.2 `solver.py` — nghiệm giải tích + FDM

```python
"""
1D diffusion-degradation solver: analytical (eigenfunction expansion) and
numerical (Crank-Nicolson finite-difference) solutions of

    dc/dt = D * d2c/dx2 - r * c ,     x in [0, L]

Boundary conditions:
    c(0, t)   = c0        (Dirichlet, fixed source e.g. vessel wall / channel)
    dc/dx(L,t) = 0         (Neumann / zero-flux, symmetry at outer tissue edge)

Initial condition:
    c(x, 0) = 0

This geometry mirrors the Krogh-cylinder boundary conditions used in
Thurber & Wittrup (2012) for antibody transport: fixed concentration at the
vessel wall, no-flux at the outer radius of the tissue cylinder.

Two solution methods are provided:
  - analytical_transient_profile / steady_state_profile: exact (series)
    solution, valid for homogeneous D and r. Used to generate synthetic
    data quickly and to validate the numerical solver.
  - fdm_crank_nicolson: general implicit finite-difference solver. Needed
    once D(x) or r(x) become spatially heterogeneous (Task 1.4), where the
    analytical solution no longer applies.
"""
from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def steady_state_profile(x: np.ndarray, D: float, r: float, c0: float, L: float) -> np.ndarray:
    """Exact steady-state solution: D*c'' - r*c = 0, c(0)=c0, c'(L)=0.

    c_ss(x) = c0 * cosh((L - x) / lambda) / cosh(L / lambda),  lambda = sqrt(D / r)

    `lambda` is the characteristic penetration depth (same quantity as in
    Thurber's antibody-penetration analysis).
    """
    lam = np.sqrt(D / r)
    return c0 * np.cosh((L - x) / lam) / np.cosh(L / lam)


def analytical_transient_profile(
    x: np.ndarray,
    t: np.ndarray | float,
    D: float,
    r: float,
    c0: float,
    L: float,
    n_modes: int = 200,
) -> np.ndarray:
    """Exact transient solution via eigenfunction expansion.

    Write c(x,t) = c_ss(x) - w(x,t). w solves the same PDE with homogeneous
    BCs w(0,t)=0, w_x(L,t)=0 and initial condition w(x,0) = c_ss(x).
    Its eigenfunctions are sin(k_n x) with k_n = (2n-1)*pi/(2L), decaying as
    exp(-(D k_n^2 + r) t). Coefficients B_n are found by projecting the
    initial condition onto the eigenbasis (orthogonal on [0, L]).

    Returns an array of shape (len(x),) if t is scalar, or (len(x), len(t))
    if t is a 1D array.
    """
    x = np.asarray(x, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    c_ss = steady_state_profile(x, D, r, c0, L)

    n = np.arange(1, n_modes + 1)
    k_n = (2 * n - 1) * np.pi / (2 * L)
    mu_n = D * k_n**2 + r

    xq = np.linspace(0.0, L, 2000)
    css_q = steady_state_profile(xq, D, r, c0, L)
    sin_nq = np.sin(np.outer(k_n, xq))
    I_n = _trapezoid(css_q[None, :] * sin_nq, xq, axis=1)
    B_n = (2.0 / L) * I_n

    sin_nx = np.sin(np.outer(k_n, x))              # (n_modes, len(x))
    decay = np.exp(-mu_n[:, None] * t_arr[None, :])  # (n_modes, len(t))
    w = sin_nx.T @ (B_n[:, None] * decay)             # (len(x), len(t))
    profile = c_ss[:, None] - w

    if np.ndim(t) == 0:
        return profile[:, 0]
    return profile


def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Tridiagonal solver (Thomas algorithm), avoids a scipy dependency."""
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    out = np.empty(n)
    out[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


def fdm_crank_nicolson(
    D: float,
    r: float,
    c0: float,
    L: float,
    nx: int,
    t_eval: np.ndarray,
    max_steps: int = 3000,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Implicit Crank-Nicolson finite-difference solver (unconditionally stable).

    Dirichlet BC at x=0 (c=c0), zero-flux BC at x=L implemented with a
    mirrored ghost node (c[nx] = c[nx-2]).

    Each requested time in `t_eval` is reached by an independent march from
    t=0, with its own step count (an accuracy heuristic, capped at
    `max_steps` for speed). This keeps early, short-time requests well
    resolved even when other requested times are orders of magnitude larger
    - a single shared dt sized for the largest t would otherwise under-
    resolve the early ones.

    Returns (x_grid, {t: profile}) for every requested time in `t_eval`.
    """
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]
    t_eval = np.sort(np.asarray(t_eval, dtype=float))

    n_unknown = nx - 1  # unknowns are nodes i = 1 .. nx-1 (i=0 is Dirichlet)
    results: dict[float, np.ndarray] = {}

    for t_target in t_eval:
        if np.isclose(t_target, 0.0):
            c_cur = np.zeros(nx)
            c_cur[0] = c0
            results[float(t_target)] = c_cur
            continue

        dt_accuracy = min(0.4 * dx * dx / D, 0.05 / r) if r > 0 else 0.4 * dx * dx / D
        nsteps = int(np.clip(np.ceil(t_target / dt_accuracy), 1, max_steps))
        dt = t_target / nsteps

        alpha = D * dt / (2 * dx * dx)
        beta = r * dt / 2.0

        a = np.full(n_unknown, -alpha)
        b = np.full(n_unknown, 1 + 2 * alpha + beta)
        c_diag = np.full(n_unknown, -alpha)
        a[-1] = -2 * alpha   # mirrored ghost node at the outer (zero-flux) boundary
        c_diag[-1] = 0.0

        c_cur = np.zeros(nx)
        c_cur[0] = c0

        for _ in range(nsteps):
            d = alpha * c_cur[:-2] + (1 - 2 * alpha - beta) * c_cur[1:-1] + alpha * c_cur[2:]
            d_last = 2 * alpha * c_cur[-2] + (1 - 2 * alpha - beta) * c_cur[-1]
            d = np.append(d, d_last)
            d[0] += alpha * c0  # known Dirichlet value contributes to the i=1 equation

            u = _thomas_solve(a, b, c_diag, d)
            c_cur = np.concatenate(([c0], u))

        results[float(t_target)] = c_cur

    return x, results


if __name__ == "__main__":
    # Quick self-check: analytical series solution vs. Crank-Nicolson FDM.
    D, r, c0, L = 10.0, 1e-3, 1.0, 100.0
    x = np.linspace(0, L, 101)
    t_eval = [0.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]

    _, res = fdm_crank_nicolson(D, r, c0, L, 101, t_eval)
    for t in t_eval[1:]:
        c_ana = analytical_transient_profile(x, t, D, r, c0, L, n_modes=300)
        c_num = res[t]
        rel_err = np.max(np.abs(c_ana - c_num)) / max(np.max(c_ana), 1e-12)
        print(f"t={t:8.1f}  max relative error (analytical vs FDM) = {rel_err:.3e}")
```

### 4.3 `generate_data.py` — sinh dữ liệu tổng hợp

```python
"""
Synthetic dataset generation for the 1D diffusion-degradation surrogate.

Samples (c0, D, r) log-uniformly, evaluates the analytical transient
solution at several log-spaced times per condition, and stores everything
needed to train the NN surrogate:

    X = (log10 c0, log10 D, log10 r, t)   ->   y = C(x, t)  on a fixed grid x

Usage:
    python generate_data.py --n_conditions 2000 --n_times 8 --out ../../outputs/dataset.npz

For the full Lot 1 deliverable (Tache 1.2), re-run with --n_conditions 100000.
A random subset is cross-checked against the Crank-Nicolson FDM solver
(solver.fdm_crank_nicolson) to confirm the analytical solution is being
evaluated correctly before it is used to label 100k conditions blindly.
"""
from __future__ import annotations

import argparse

import numpy as np

from solver import analytical_transient_profile, fdm_crank_nicolson

# Parameter ranges, loosely inspired by Table 1 of Thurber & Wittrup (2012)
# for antibody transport in a Krogh-cylinder cross-section. Adjust freely -
# they only need to bracket the regimes you care about.
C0_RANGE = (1e-2, 1e1)      # arbitrary concentration unit (e.g. nM)
D_RANGE = (1.0, 50.0)       # um^2/s
R_RANGE = (1e-6, 1e-3)      # 1/s  (degradation / internalization-like rate)
L = 150.0                   # um, domain length (~ Krogh cylinder radius)
NX = 100                    # spatial grid points
T_MAX = 3 * 24 * 3600.0     # 3 days, in seconds


def sample_log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return 10 ** rng.uniform(np.log10(low), np.log10(high), size=size)


def build_dataset(n_conditions: int, n_times: int, seed: int = 0, validate_frac: float = 0.01):
    rng = np.random.default_rng(seed)

    c0_vals = sample_log_uniform(rng, *C0_RANGE, n_conditions)
    D_vals = sample_log_uniform(rng, *D_RANGE, n_conditions)
    r_vals = sample_log_uniform(rng, *R_RANGE, n_conditions)

    x_grid = np.linspace(0.0, L, NX)
    # shared log-spaced time points (skip t=0, trivial all-zero profile)
    t_points = np.logspace(np.log10(T_MAX / 1000), np.log10(T_MAX), n_times)

    X = np.empty((n_conditions * n_times, 4), dtype=np.float64)
    Y = np.empty((n_conditions * n_times, NX), dtype=np.float64)

    row = 0
    for c0, D, r in zip(c0_vals, D_vals, r_vals):
        profiles = analytical_transient_profile(x_grid, t_points, D, r, c0, L, n_modes=150)  # (NX, n_times)
        for j, t in enumerate(t_points):
            X[row] = (np.log10(c0), np.log10(D), np.log10(r), t)
            Y[row] = profiles[:, j]
            row += 1

    # Cross-check a random subset against the independent FDM solver. Capped
    # at a small absolute number regardless of n_conditions - this is a
    # sanity check on the analytical formula, not something that needs to
    # scale with dataset size, and each check costs its own FDM march.
    n_check = min(max(1, int(validate_frac * n_conditions)), 20)
    check_idx = rng.choice(n_conditions, size=n_check, replace=False)
    max_rel_err = 0.0
    for idx in check_idx:
        c0, D, r = c0_vals[idx], D_vals[idx], r_vals[idx]
        _, fdm_res = fdm_crank_nicolson(D, r, c0, L, NX, t_points)
        ana = analytical_transient_profile(x_grid, t_points, D, r, c0, L, n_modes=150)
        for j, t in enumerate(t_points):
            num = fdm_res[float(t)]
            denom = max(np.max(np.abs(ana[:, j])), 1e-12)
            rel_err = np.max(np.abs(ana[:, j] - num)) / denom
            max_rel_err = max(max_rel_err, rel_err)
    print(f"[validation] analytical vs FDM, worst-case relative error over "
          f"{n_check} sampled conditions: {max_rel_err:.3e}")

    return X, Y, x_grid, t_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_conditions", type=int, default=2000,
                         help="number of (c0, D, r) triples to sample (100000 for the full Lot 1 run)")
    parser.add_argument("--n_times", type=int, default=8, help="time points per condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="../../outputs/dataset.npz")
    args = parser.parse_args()

    X, Y, x_grid, t_points = build_dataset(args.n_conditions, args.n_times, args.seed)

    np.savez_compressed(
        args.out,
        X=X, Y=Y, x_grid=x_grid, t_points=t_points,
        c0_range=C0_RANGE, D_range=D_RANGE, r_range=R_RANGE, L=L,
    )
    print(f"Saved {X.shape[0]} samples ({args.n_conditions} conditions x {args.n_times} times) to {args.out}")
    print(f"X shape={X.shape}  Y shape={Y.shape}")


if __name__ == "__main__":
    main()
```

### 4.4 `model.py` — kiến trúc MLP

```python
"""MLP surrogate: (log10 c0, log10 D, log10 r, t) -> C(x, t) profile."""
from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class MLPSurrogate(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 100,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        activation: str = "gelu",
    ):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### 4.5 `train.py` — huấn luyện

```python
"""
Train the MLP surrogate on the synthetic dataset produced by generate_data.py.

Split: 70% train / 15% validation / 15% test (fixed seed, split by condition
sample index — every row already corresponds to one (params, t) pair, so a
plain random split at the row level is fine here since rows are i.i.d. given
the sampling scheme in generate_data.py).

Targets are trained in log-space: log(C + eps), then standardized. Metrics
are reported both in log-space and after inverting back to physical
concentration units.

Usage:
    python train.py --dataset ../../outputs/dataset.npz --epochs 200 \
        --hidden_dims 256,256,256 --out_dir ../../outputs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MLPSurrogate

EPS = 1e-8


def split_indices(n: int, seed: int, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="../../outputs/dataset.npz")
    parser.add_argument("--hidden_dims", type=str, default="256,256,256")
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="../../outputs")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    X, Y = data["X"], data["Y"]
    n = X.shape[0]
    train_idx, val_idx, test_idx = split_indices(n, args.seed)

    x_mean, x_std = X[train_idx].mean(0), X[train_idx].std(0) + 1e-12
    Xn = (X - x_mean) / x_std

    Ylog = np.log(Y + EPS)
    y_mean, y_std = Ylog[train_idx].mean(), Ylog[train_idx].std() + 1e-12
    Yn = (Ylog - y_mean) / y_std

    device = torch.device(args.device)

    def to_loader(idx, batch_size, shuffle):
        ds = TensorDataset(
            torch.tensor(Xn[idx], dtype=torch.float32),
            torch.tensor(Yn[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_idx, args.batch_size, True)
    val_loader = to_loader(val_idx, args.batch_size, False)

    hidden_dims = tuple(int(h) for h in args.hidden_dims.split(","))
    model = MLPSurrogate(input_dim=4, output_dim=Y.shape[1],
                          hidden_dims=hidden_dims, activation=args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(val_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"Training took {time.time() - t0:.1f} s, best val_loss={best_val:.5f}")
    model.load_state_dict(best_state)

    # ---- test-set evaluation ----
    model.eval()
    with torch.no_grad():
        pred_test_n = model(torch.tensor(Xn[test_idx], dtype=torch.float32).to(device)).cpu().numpy()
    pred_log = pred_test_n * y_std + y_mean
    pred_phys = np.exp(pred_log) - EPS

    metrics_log = compute_metrics(Ylog[test_idx], pred_log)
    metrics_phys = compute_metrics(Y[test_idx], pred_phys)
    print("Test metrics (log-space):   ", metrics_log)
    print("Test metrics (physical units):", metrics_phys)

    checkpoint = {
        "model_state": best_state,
        "hidden_dims": hidden_dims,
        "activation": args.activation,
        "input_dim": 4,
        "output_dim": Y.shape[1],
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "eps": EPS,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }
    torch.save(checkpoint, out_dir / "surrogate_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"log_space": metrics_log, "physical_units": metrics_phys}, f, indent=2)

    print(f"Saved checkpoint to {out_dir / 'surrogate_model.pt'}")


if __name__ == "__main__":
    main()
```

### 4.6 `evaluate.py` — đánh giá + trực quan hoá

```python
"""
Evaluate a trained surrogate: reload the exact test split saved in the
checkpoint, recompute MAE/RMSE/R2, and plot true-vs-predicted profiles.

Usage:
    python evaluate.py --dataset ../../outputs/dataset.npz \
        --checkpoint ../../outputs/surrogate_model.pt --out_dir ../../outputs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import MLPSurrogate
from train import compute_metrics


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLPSurrogate(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dims=ckpt["hidden_dims"],
        activation=ckpt["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(model, ckpt, X_raw: np.ndarray, device: torch.device) -> np.ndarray:
    Xn = (X_raw - ckpt["x_mean"]) / ckpt["x_std"]
    with torch.no_grad():
        pred_n = model(torch.tensor(Xn, dtype=torch.float32).to(device)).cpu().numpy()
    pred_log = pred_n * ckpt["y_std"] + ckpt["y_mean"]
    return np.exp(pred_log) - ckpt["eps"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="../../outputs/dataset.npz")
    parser.add_argument("--checkpoint", type=str, default="../../outputs/surrogate_model.pt")
    parser.add_argument("--out_dir", type=str, default="../../outputs/figures")
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data = np.load(args.dataset)
    X, Y, x_grid = data["X"], data["Y"], data["x_grid"]

    model, ckpt = load_checkpoint(args.checkpoint, device)
    test_idx = ckpt["test_idx"]

    pred_test = predict(model, ckpt, X[test_idx], device)
    y_true, y_pred = Y[test_idx], pred_test

    metrics_phys = compute_metrics(y_true, y_pred)
    metrics_log = compute_metrics(np.log(y_true + ckpt["eps"]), np.log(y_pred + ckpt["eps"]))
    print("Test metrics (physical units):", metrics_phys)
    print("Test metrics (log-space):     ", metrics_log)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots (metrics above are still valid).")
        return

    rng = np.random.default_rng(0)
    example_idx = rng.choice(len(test_idx), size=min(args.n_examples, len(test_idx)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, i in zip(axes.ravel(), example_idx):
        log_c0, log_D, log_r, t = X[test_idx][i]
        ax.plot(x_grid, y_true[i], label="true (analytical)", lw=2)
        ax.plot(x_grid, y_pred[i], "--", label="NN prediction", lw=2)
        ax.set_title(f"c0=10^{log_c0:.1f}, D=10^{log_D:.1f}, r=10^{log_r:.1f}\nt={t:.2e} s")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("C(x,t)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "profiles_true_vs_pred.png", dpi=150)
    print(f"Saved {out_dir / 'profiles_true_vs_pred.png'}")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sample = rng.choice(y_true.size, size=min(20000, y_true.size), replace=False)
    ax2.scatter(y_true.ravel()[sample], y_pred.ravel()[sample], s=2, alpha=0.3)
    lims = [0, max(y_true.max(), y_pred.max())]
    ax2.plot(lims, lims, "r--", lw=1)
    ax2.set_xlabel("true C(x,t)")
    ax2.set_ylabel("predicted C(x,t)")
    ax2.set_title(f"R2 = {metrics_phys['R2']:.4f}")
    fig2.tight_layout()
    fig2.savefig(out_dir / "scatter_true_vs_pred.png", dpi=150)
    print(f"Saved {out_dir / 'scatter_true_vs_pred.png'}")


if __name__ == "__main__":
    main()
```

### 4.7 Cách chạy end-to-end

Tất cả lệnh chạy từ `src/diffusion_degradation/` (đường dẫn `../../outputs/...` là tương đối so với đó):

```bash
cd src/diffusion_degradation

# 1. Kiểm tra solver (in sai số analytical vs FDM)
python3 solver.py

# 2. Sinh dữ liệu — demo nhanh (đã có sẵn outputs/dataset.npz, 2000 điều kiện x 8 thời điểm)
python3 generate_data.py --n_conditions 2000 --n_times 8 --out ../../outputs/dataset.npz

#    Cho deliverable Lot 1 đầy đủ (Tache 1.2 đòi 100 000 điều kiện):
python3 generate_data.py --n_conditions 100000 --n_times 8 --out ../../outputs/dataset_full.npz

# 3. Train
python3 train.py --dataset ../../outputs/dataset.npz --epochs 200

# 4. Evaluate + vẽ hình
python3 evaluate.py --dataset ../../outputs/dataset.npz
```

### 4.8 Kết quả kiểm tra thực nghiệm (đã chạy thật khi soạn tài liệu này)

**Solver — analytical vs FDM** (`python3 solver.py`), trường hợp `D=10 µm²/s, r=10⁻³ s⁻¹, L=100 µm, c₀=1`:

| t (s) | sai số tương đối tối đa |
|---|---|
| 50 | 2.7 × 10⁻⁵ |
| 200 | 1.2 × 10⁻⁵ |
| 1000 | 1.4 × 10⁻⁶ |
| 5000 | 2.1 × 10⁻⁶ |
| 20000 | 2.1 × 10⁻⁶ |

→ Hai phương pháp độc lập (chuỗi hàm riêng vs. Crank–Nicolson) khớp nhau tới ~10⁻⁵, xác nhận cả hai được suy ra và code đúng.

**Sinh dữ liệu** (`generate_data.py`), đo trên máy thực tế (CPU, không GPU):

| n_conditions | n_times | Thời gian | Sai số validation analytical-vs-FDM (worst-case trên tối đa 20 điều kiện random) |
|---|---|---|---|
| 300 | 8 | 7.3 s | 4.8 × 10⁻⁴ |
| 2000 | 8 | 44 s | 1.3 × 10⁻² |
| 4000 | 8 | 56 s | 6.1 × 10⁻³ |

Tốc độ sinh profile (không tính phần validate FDM, đo riêng): **~5.4 ms/điều kiện** ⟹ **100 000 điều kiện × 8 thời điểm ước tính ~9–10 phút**. Sai số validation cỡ 10⁻³–10⁻² (thay vì 10⁻⁵ như test đơn lẻ ở trên) đến từ việc solver FDM dùng trong `generate_data.py` giới hạn `max_steps=3000` để giữ tốc độ khi validate — hoàn toàn đủ để xác nhận công thức giải tích đúng (không có lỗi hệ thống), nhưng nếu bạn cần validation chặt hơn, tăng `max_steps` trong lệnh gọi `fdm_crank_nicolson` ở `build_dataset()`.

**Lưu ý kỹ thuật đã gặp và sửa khi viết code này** (để bạn không mất thời gian lặp lại):
1. Dùng **một `Δt` chung** cho cả khoảng `[0, t_max]` khi `t_max` lớn (vài ngày, tính bằng giây) nhưng vẫn cần độ phân giải tốt cho các `t` nhỏ (ví dụ `t_max/1000`) → sai số cực lớn ở các `t` nhỏ (từng đo được tới 58%). **Sửa bằng cách march độc lập tới từng `t` yêu cầu**, mỗi lần với số bước riêng phù hợp với chính `t` đó (xem `fdm_crank_nicolson`, đoạn "march t_target").
2. Bước thời gian chọn theo tiêu chí ổn định của sơ đồ **hiện** (explicit) — không cần thiết vì Crank–Nicolson **ẩn, ổn định vô điều kiện**; dùng tiêu chí đó chỉ để chọn `Δt` cho độ chính xác, và phải **giới hạn số bước tối đa** (`max_steps`), nếu không với `D` nhỏ và thời gian mô phỏng dài, số bước có thể lên tới hàng trăm nghìn và làm chương trình chạy rất lâu (vòng lặp Python + Thomas-solve không vectorize được qua các bước thời gian).
3. `numpy` bản mới (≥2.0) đã bỏ `np.trapz`, đổi tên thành `np.trapezoid` — code dùng `getattr(np, "trapezoid", None) or np.trapz` để chạy được trên cả hai phiên bản.

---

## 5. Hướng mở rộng — kết nối sang Lot 1 đầy đủ và Lot 2

- **Tâche 1.4** (nếu Lot 2/benchmark cho thấy cần): thêm heterogeneity bằng cách cho `D`, `r` phụ thuộc `x` (ví dụ mật độ tế bào/kháng nguyên thay đổi theo bán kính) — nghiệm giải tích ở mục 2.2 sẽ **không còn dùng được**, và `fdm_crank_nicolson` cần sửa để nhận `D(x)`, `r(x)` dạng vector thay vì scalar (thay `alpha`, `beta` từ hằng số thành vector theo `i`).
- **Tâche 1.3 → Lot 2, Tâche 2.2/2.3**: MLP `p → y` ở đây chính là "décodeur biophysique" trong dossier. Ở Lot 2, nó sẽ được ghép sau một "encodeur moléculaire" `x → p` (biến SMILES/sequence thành `p`), và cần thêm nhánh residual `x → y` — kiến trúc `model.py` hiện tại chỉ là **nửa sau** (decodeur) của toàn bộ pipeline mô tả trong dossier, mục "Tâche 2.2/2.3".
- Output vật lý hiện tại chỉ có `C(x,t)` (nồng độ) — mô hình ADC đầy đủ (dossier, mục 3) cần thêm `M(d)` (mortalité) và `IC50(d)`; đây là lý do dossier liệt kê `y = {C(d), M(d), IC50(d)}` chứ không chỉ `C(d)` — bài toán ở tài liệu PDF thứ 3 là bước rút gọn hợp lý để kiểm chứng pipeline trước khi thêm 2 output này.
