# Lộ trình học nền tảng — Syntopia / PRISMA-ADC

**Dự án:** Syntopia — PRISMA-ADC (*PRédiction par IA et Simulation des Mécanismes d'Action des ADC*)
**Vai trò của bạn:** bắt đầu từ Lot 1 (simulateur biophysique + décodeur IA), mở rộng dần sang Lot 2/3.
**Tài liệu này trả lời:** *Tôi cần học nền tảng gì? Học theo trình tự nào? Đọc/tham khảo ở đâu?*

Nguồn gốc: tổng hợp từ 3 papers trong `papers/` và dossier dự án:

- **[Thurber 2012]** — *A mechanistic compartmental model for total antibody uptake in tumors* (`2012 Thurber cylinder JTB.pdf`). Mô hình vật lý nền tảng: khuếch tán–phản ứng quanh mạch máu (Krogh cylinder).
- **[NN-surrogate]** — *Neural-Network Surrogate Model for 1D Diffusion-Degradation* (`NN_surrogate_diffusion_degradation.pdf`). Bài khởi động (proof of concept) — đã được triển khai trong `src/diffusion_degradation/`.
- **[Dossier]** — *Dossier Pionniers de l'IA* (bản trình bày dự án). Cho biết bức tranh lớn: encodeur phân tử → décodeur biophysique → hiệu quả không gian, apprentissage actif.

> Tài liệu đồng hành: [HUONG_DAN_PROJECT.md](HUONG_DAN_PROJECT.md) giải thích chi tiết lý thuyết + code của bài 1D. Đọc tài liệu này **trước** để có bản đồ, rồi dùng HUONG_DAN_PROJECT.md khi đi vào chi tiết Giai đoạn 2.

---

## 1. Bức tranh lớn: dự án đòi hỏi giao của 3 lĩnh vực

Để đóng góp được cho PRISMA-ADC, bạn cần đứng ở giao điểm của ba mảng kiến thức. Không cần giỏi cả ba ngay — cần đủ để nối được chúng với nhau.

| Mảng | Bạn cần hiểu đến mức | Vì sao dự án cần |
|---|---|---|
| **A. Sinh học ADC & vận chuyển thuốc trong u** | Khái niệm + trực giác cơ chế | Biết mình đang mô phỏng cái gì, tại sao "binding site barrier" và "bystander effect" quan trọng |
| **B. Toán/vật lý: PDE khuếch tán–phản ứng + giải số** | Làm được, viết được solver | Là "simulateur biophysique" sinh dữ liệu tổng hợp (Lot 1) |
| **C. Machine Learning: surrogate model, encodeur phân tử, active learning** | Làm được, train được | Là "décodeur IA" học từ dữ liệu (Lot 1→3) |

Chiến lược của dự án (từ [Dossier], mục *Méthodologie*): **dùng vật lý (B) để sinh dữ liệu tổng hợp → pre-train mô hình IA (C) → gắn với mô tả phân tử (C) → tăng tốc bằng active learning (C)**, trong bối cảnh sinh học (A). Ba mảng nối theo đúng thứ tự đó, nên lộ trình học cũng đi theo thứ tự đó.

---

## 2. Kiến thức nền tảng cần có (checklist)

### A. Sinh học & dược động học ADC — *mức khái niệm*

- [ ] **Kháng thể (antibody / IgG)** là gì, kích thước lớn (~150 kDa) → khuếch tán chậm, thấm mô kém.
- [ ] **ADC = Antibody-Drug Conjugate**: kháng thể (nhắm đích) + linker + payload (thuốc độc tế bào). 5 tham số thiết kế: dose, payload, linker, **DAR** (drug-to-antibody ratio), site conjugaison.
- [ ] **Antigen / thụ thể** (ví dụ HER2): kháng thể gắn vào → nội hoá (internalization) → giải phóng payload.
- [ ] Bốn hiện tượng không gian quyết định hiệu quả (từ [Dossier], mục a):
  - **Diffusion** trong ma trận mô.
  - **Binding site barrier**: kháng thể bị "bắt" hết ở rìa u (bão hoà thụ thể) → không vào sâu được.
  - **Internalisation** + giải phóng payload.
  - **Bystander effect**: payload lan sang tế bào lân cận.
- [ ] Vì sao mô hình động vật (PDX) và nuôi cấy 2D không đủ → cần mô phỏng + dữ liệu 3D.

> **Không cần** học sâu miễn dịch học/hoá dược. Cần đủ để đọc [Thurber 2012] mà không lạc.

### B. Toán & vật lý mô phỏng — *mức làm được*

- [ ] **Giải tích/ĐSTT cơ bản**: đạo hàm riêng, ma trận, hệ tuyến tính (bạn sẽ giải `A·c = b`).
- [ ] **Phương trình khuếch tán–phản ứng (reaction–diffusion PDE)**:
  `∂c/∂t = D ∂²c/∂x² − r·c`
  - Ý nghĩa từng số hạng; **độ dài thấm** `λ = √(D/r)` (thang chiều dài then chốt).
- [ ] **Điều kiện biên**: Dirichlet (`c(0,t)=c₀`, thành mạch) và Neumann/zero-flux (`∂c/∂x|_L=0`, biên đối xứng Krogh).
- [ ] **Nghiệm giải tích** cho PDE tuyến tính (steady-state + tách biến / eigenfunction) — dùng làm "ground truth" kiểm tra solver.
- [ ] **Giải số PDE**:
  - Sai phân hữu hạn (FDM), rời rạc hoá Laplacian.
  - Sơ đồ thời gian: explicit (điều kiện ổn định CFL) vs implicit vs **Crank–Nicolson** (dùng trong `solver.py`).
  - Khái niệm về phần tử hữu hạn (FEM) — dossier nhắc "éléments finis"; đủ hiểu ý tưởng.
- [ ] **Mô hình Krogh cylinder** ([Thurber 2012] Fig. 1): hình học trụ, khuếch tán bán kính từ mạch máu; **số Biot** (`Bi = 2·P·R_cap/(D·ε)`) — vì sao `Bi` nhỏ cho phép rút bài 3D về mô hình "0 chiều" (compartmental).

### C. Machine Learning — *mức làm được*

- [ ] **Nền tảng**: train/val/test split, overfitting, loss, gradient descent, learning rate.
- [ ] **MLP (Multi-Layer Perceptron)**: fully-connected, ReLU/GELU, số layer/units. Kiến trúc surrogate trong [NN-surrogate]: input 4, 3–5 hidden, 128–256 units, output = số điểm lưới.
- [ ] **Surrogate modeling / emulator**: ý tưởng "thay solver chậm bằng NN nhanh"; học ánh xạ `tham số vật lý p → profile y`.
- [ ] **Kỹ thuật huấn luyện**: chuẩn hoá input (log-scale cho `c₀,D,r`), log-transform output `log(C+ε)`, chỉ số đánh giá **MAE / RMSE / R²**.
- [ ] **PyTorch**: Dataset/DataLoader, `nn.Module`, training loop, lưu/nạp model.
- [ ] **Biểu diễn phân tử** (cho Lot 2, học sau): SMILES; fingerprint **ECFP**; descriptor **RDKit / Mordred**; embedding từ mô hình nền (protein/ligand). Encodeur `x → p`.
- [ ] **PINN / Universal Differential Equations** — *chỉ cần biết là gì và vì sao dossier chọn KHÔNG dùng chúng* (chọn surrogate offline thay vì PINN tích hợp). Đọc [Raissi 2019] ở mức khái quát.
- [ ] **Active learning**: acquisition function, exploration (uncertainty — Monte Carlo Dropout) vs exploitation; vì sao giảm được số thí nghiệm (Lot 3).

---

## 3. Quy trình học theo giai đoạn (roadmap)

Bốn giai đoạn, bám sát cấu trúc Lot 1 → Lot 3 của dự án. Ước lượng thời gian giả định ~15–20h/tuần; điều chỉnh theo bạn.

### Giai đoạn 0 — Định hướng (≈ 3–4 ngày)

**Mục tiêu:** hiểu *tại sao* dự án tồn tại và bản đồ tổng thể.

1. Đọc [Dossier] — mục 2 (Problème/État de l'art) và mục 3 (Objectifs, Méthodologie, Lot 1–3). Không cần hiểu hết chi tiết kỹ thuật; nắm mạch logic: *dữ liệu tổng hợp → pre-train → encodeur phân tử → active learning*.
2. Đọc phần **Abstract + Introduction** của [Thurber 2012]. Trả lời được: vì sao kháng thể thấm vào u chậm? "extravasation" là bước giới hạn nghĩa là gì?
3. Đọc [NN-surrogate] (chỉ 1 trang) và mục 1 của [HUONG_DAN_PROJECT.md]. Hiểu bài 1D là *proof of concept rút gọn* của Lot 1.
4. **Kiểm tra hiểu:** tự viết 1 đoạn ngắn nối 3 papers: papers nào là vật lý nền, papers nào là bản rút gọn của việc bạn sắp làm, papers nào là bối cảnh.

**Xong khi:** giải thích được cho người khác "PRISMA-ADC làm gì và Lot 1 khớp vào đâu".

### Giai đoạn 1 — Vật lý & giải số PDE (≈ 1.5–2 tuần)

**Mục tiêu:** hiểu và tự tay chạy được `simulateur biophysique` 1D.

1. **Ôn PDE khuếch tán–phản ứng**: ý nghĩa vật lý, `λ=√(D/r)`, vai trò `c₀` chỉ là hệ số nhân tuyến tính.
2. **Đọc kỹ [Thurber 2012] mục 2 (Methods)** + Fig. 1: binding, vascular transport, internalization; các phương trình (1)–(6); Table 1 (dải giá trị `D, k_e, [Ag], K_d…` — bạn sẽ dùng làm dải sampling).
3. **Nghiệm giải tích**: đọc [HUONG_DAN_PROJECT.md] mục 2.2; tự dẫn lại nghiệm steady-state `D c'' − r c = 0`.
4. **Giải số**: học FDM + Crank–Nicolson; đọc `src/diffusion_degradation/solver.py` **dòng theo dòng**, đối chiếu với công thức.
5. **Bài tập tay-lấm-bẩn:**
   - Chạy `generate_data.py`, `train.py`, `evaluate.py` (xem mục "Chạy" bên dưới).
   - Đổi `D` và `r`, quan sát độ dài thấm thay đổi; xác nhận `λ=√(D/r)` khớp hình.
   - So nghiệm solver với nghiệm giải tích trên vài ca → sai số phải rất nhỏ.

**Xong khi:** giải thích được từng dòng `solver.py` và dự đoán được hình dạng profile khi đổi `D, r`.

### Giai đoạn 2 — Surrogate ML cho bài 1D (≈ 1.5–2 tuần)

**Mục tiêu:** hiểu và cải thiện được `décodeur biophysique` (MLP `p → y`).

1. **Ôn ML cơ bản + PyTorch** nếu chưa vững (xem tham khảo mục 4.C).
2. Đọc `model.py` + `train.py` + `evaluate.py`; nắm: log-transform input/output, split 70/15/15, tính MAE/RMSE/R².
3. **Thí nghiệm học tập:**
   - Đổi số layer/units, activation ReLU↔GELU; đo lại RMSE/R².
   - Vẽ true-vs-predicted ở vài thời điểm `t`; tìm vùng lỗi lớn (thường t nhỏ, gradient dốc).
   - Thử bỏ log-transform output → xem sai số tệ đi thế nào (hiểu *vì sao* cần nó).
4. **Kết nối lý thuyết:** vì sao NN học được ánh xạ này dễ (PDE tuyến tính, profile trơn)? Ghi lại — vì bài ADC thật *phi tuyến* (bão hoà thụ thể) sẽ khó hơn.

**Xong khi:** tự cải thiện được một chỉ số của surrogate và giải thích được đánh đổi kiến trúc.

### Giai đoạn 3 — Mở rộng sang ADC thật & IA đầy đủ (≈ 2–4 tuần, liên tục)

**Mục tiêu:** sẵn sàng đóng góp cho Lot 1 (đầy đủ) và Lot 2.

1. **Từ 1D → ADC multiscale:** đọc [Cilliers 2016] (ref [2] của dossier). Thêm biến: nội hoá, giải phóng payload, bystander. Đây là bước từ `∂c/∂t = D∇²c − r c` sang hệ nhiều phương trình phản ứng.
2. **Đọc lại [Thurber 2012] mục 3 (Results) + sensitivity (mục 3.5):** tham số nào quan trọng, tham số nào bỏ được — trực giác này định hướng thiết kế simulateur đầy đủ.
3. **Biểu diễn phân tử (encodeur `x→p`):** học SMILES, ECFP, RDKit/Mordred; chạy thử tính descriptor cho vài payload/linker.
4. **Kiến trúc end-to-end** ([Dossier] Tâche 2.3): encodeur `x→p` + décodeur `p→y` + kết nối dư `x→y`. Hiểu vì sao tách hai khối (pre-train décodeur bằng dữ liệu tổng hợp rồi mới nối encodeur).
5. **Active learning** ([Dossier] Lot 3): đọc [Reker & Schneider 2015]; hiểu Monte Carlo Dropout để ước lượng bất định; ý tưởng chọn thí nghiệm thông tin nhất.
6. **(Đọc để biết)** [Raissi 2019] PINN — để hiểu dossier *đã cân nhắc và loại* hướng này, và lập luận đánh đổi.

**Xong khi:** phác thảo được cách mở rộng simulateur 1D hiện tại thành mô-đun ADC, và mô tả được pipeline IA đầy đủ.

---

## 4. Tham khảo theo chủ đề

### A. Sinh học ADC & vận chuyển trong u
- **[Thurber 2012]** `papers/2012 Thurber cylinder JTB.pdf` — *đọc chính*. Mô hình compartmental, Krogh cylinder, Table 1 (tham số).
- **[Cilliers 2016]** Cilliers C. et al., *Multiscale Modeling of Antibody-Drug Conjugates*, AAPS J. 2016;18(5):1117-1130 (ref [2] dossier) — mở rộng ADC.
- **[Thurber 2008]** Thurber, Schmidt, Wittrup, *Antibody tumor penetration: transport opposed by systemic and antigen-mediated clearance*, Adv Drug Deliv Rev. 2008;60(12):1421-34 (ref [1] dossier) — review nền tảng, khái niệm binding-site barrier.
- Nền tảng ADC (tổng quan): tìm review "antibody-drug conjugate mechanism DAR bystander" — bất kỳ review gần đây trên Nature Reviews Drug Discovery.

### B. PDE & giải số
- Reaction–diffusion, độ dài thấm: bất kỳ giáo trình transport phenomena (Bird, Stewart, Lightfoot) hoặc notes online "reaction-diffusion penetration depth".
- FDM / Crank–Nicolson: LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations* (chương parabolic PDE). Hoặc notes MIT 18.336 / bất kỳ tutorial "Crank-Nicolson heat equation Python".
- Krogh cylinder & số Biot: [Thurber 2012] mục 2 là nguồn trực tiếp và đủ.
- Code tham chiếu sẵn có: `src/diffusion_degradation/solver.py` (+ giải thích ở [HUONG_DAN_PROJECT.md] mục 2–3).

### C. Machine Learning
- PyTorch: tutorial chính thức "Deep Learning with PyTorch: A 60 Minute Blitz" + "Learn the Basics".
- MLP / nền tảng: 3Blue1Brown *Neural Networks* (trực giác); *Dive into Deep Learning* (d2l.ai) chương MLP.
- Surrogate / emulator: khái niệm — tìm "neural network surrogate model PDE emulator".
- Biểu diễn phân tử: tài liệu **RDKit** (getting started), **Mordred** descriptors, khái niệm **ECFP** (Rogers & Hahn 2010, *Extended-Connectivity Fingerprints*).
- **[Raissi 2019]** Physics-Informed Neural Networks, J Comput Phys 378:686-707 (ref [3] dossier) — *đọc khái quát*.
- **[Reker & Schneider 2015]** *Active-learning strategies in computer-assisted drug discovery*, Drug Discov Today 20(4):458-465 (ref [4] dossier).
- Monte Carlo Dropout (uncertainty): Gal & Ghahramani 2016, *Dropout as a Bayesian Approximation*.

### D. Bối cảnh dự án
- **[Dossier]** `papers/Dossier Pionniers de l'IA...pdf` — mục 2, 3, và các Lot.
- **[HUONG_DAN_PROJECT.md](HUONG_DAN_PROJECT.md)** — hướng dẫn chi tiết bài 1D (lý thuyết + code + số liệu chạy thật).

---

## 5. Chạy code tham chiếu (Giai đoạn 1–2)

Code đã có sẵn và đã chạy thật. Trình tự:

```bash
cd /home/duc-khiem/Documents/SummerInternship/SYNTOPIA
pip install -r requirements.txt          # numpy, torch, matplotlib

python -m src.diffusion_degradation.generate_data   # sinh outputs/dataset.npz
python -m src.diffusion_degradation.train           # train MLP, in MAE/RMSE/R²
python -m src.diffusion_degradation.evaluate        # đánh giá + vẽ hình so sánh
```

Đọc kèm mỗi file theo Giai đoạn 1–2 ở trên. Nếu lệnh chạy khác với `-m`, mở từng file xem `__main__`.

---

## 6. Tự kiểm tra: bạn đã sẵn sàng khi trả lời được

1. Vì sao kháng thể/ADC thấm vào u chậm, và bước nào là giới hạn tốc độ? *(A, [Thurber 2012])*
2. Trong `∂c/∂t = D∂²c/∂x² − r c`, tăng `r` thì profile thay đổi ra sao? Độ dài thấm là gì? *(B)*
3. Vì sao dùng Crank–Nicolson thay vì explicit Euler? *(B, `solver.py`)*
4. Vì sao log-transform input `(c₀,D,r)` và output `C`? *(C, `train.py`)*
5. Surrogate MLP thay thế cái gì, và lợi ích tốc độ ở đâu trong pipeline dự án? *(C, [Dossier])*
6. Bài 1D tuyến tính khác bài ADC thật ở điểm cốt lõi nào (gợi ý: binding-site barrier → phi tuyến)? *(A+B)*
7. Vì sao dossier chọn surrogate offline thay vì PINN? *(C, [Raissi 2019] + Dossier)*
8. Active learning giúp giảm số thí nghiệm bằng cách nào? *(C, [Reker 2015], Lot 3)*

Trả lời trơn tru cả 8 câu ⇒ bạn đã nắm nền tảng để bắt tay vào Lot 1 đầy đủ.
