%% thurber_model_visualize.m
%
% Mo phong va truc quan hoa nghiem giai tich cua mo hinh compartmental
% Thurber & Wittrup (2012), "A mechanistic compartmental model for total
% antibody uptake in tumors", J. Theor. Biol. 314, 57-68.
%
% Phuong trinh chu dao (Eq. 6 trong bai bao):
%
%   d[Ab]_total/dt = kex * [Ab]_plasma(t) * Kd/(Ag_eps + Kd)
%                    - ke * Ag_eps/(Ag_eps + Kd) * [Ab]_total
%
% voi:
%   kex     = 2*P*Rcap/RKrogh^2   toc do thoat mach (extravasation rate)
%   ke                          toc do noi hoa + thoai hoa khang the
%   Kd                          hang so phan ly khang the-khang nguyen
%   Ag_eps  = [Ag]/epsilon        nong do khang nguyen hieu dung trong mo
%   [Ab]_plasma(t) = [Ab]_plasma0*(A*exp(-ka*t) + B*exp(-kb*t))  (2 pha)
%
% Nghiem giai tich dang dong (Eq. 7-8):
%
%   [Ab]_total(t)/[Ab]_plasma0 = kex * ( ...
%         A/(Omega-ka) * (exp(-ka*t) - exp(-Omega*t)) + ...
%         B/(Omega-kb) * (exp(-kb*t) - exp(-Omega*t)) )
%
%   Omega = kex*Kd/(Ag_eps+Kd) + ke*Ag_eps/(Ag_eps+Kd)
%
% Ket qua duoc doi sang %ID/g (Eq. 10):
%   %ID/g = 100 * ([Ab]_total/[Ab]_plasma0) / (Vplasma * rho)

clear; clc; close all;

%% --- Tham so co dinh (gia tri dien hinh, Table 1-2 cua bai bao) ---
kex = 0.737;    % [1/ngay] toc do thoat mach, gia tri fit cho IgG (Table 2)
ke  = 1.109;    % [1/ngay] toc do noi hoa+thoai hoa (tuong duong t1/2 = 15h)

% Duoc dong hoc huyet tuong hai pha, dien hinh cho IgG chuot
A_frac = 0.6;  ka = 5.00;   % pha alpha: phan bo nhanh
B_frac = 0.4;  kb = 0.05;   % pha beta: thai truong (t1/2 ~ 14 ngay)

Vplasma = 2;    % [mL] the tich huyet tuong chuot
rho     = 1;    % [g/mL] mat do mo khoi u

t = linspace(0, 14, 500);   % truc thoi gian: 0 -> 14 ngay

%% --- Ham tinh Omega (Eq. 8) va ty le [Ab]total/[Ab]plasma0 (Eq. 7) ---
Omega_fun = @(Kd, Ag_eps) kex.*Kd./(Ag_eps+Kd) + ke.*Ag_eps./(Ag_eps+Kd);

Ab_ratio = @(t, Kd, Ag_eps) kex .* ( ...
    A_frac ./ (Omega_fun(Kd,Ag_eps)-ka) .* (exp(-ka.*t) - exp(-Omega_fun(Kd,Ag_eps).*t)) + ...
    B_frac ./ (Omega_fun(Kd,Ag_eps)-kb) .* (exp(-kb.*t) - exp(-Omega_fun(Kd,Ag_eps).*t)) );

IDg_fun = @(t, Kd, Ag_eps) 100 * Ab_ratio(t, Kd, Ag_eps) / (Vplasma*rho);

%% --- Kich ban 1: anh huong cua ai luc Kd (binding site barrier) ---
Ag_eps_fixed = 100;                  % [nM] muc bieu hien khang nguyen co dinh
Kd_list      = [0.1, 1, 10, 100];    % [nM] tu ai luc rat cao den thap
colors1      = lines(numel(Kd_list));

figure('Position', [100 100 900 750]);

subplot(2,1,1); hold on; box on;
for i = 1:numel(Kd_list)
    plot(t, IDg_fun(t, Kd_list(i), Ag_eps_fixed), 'LineWidth', 2, ...
        'Color', colors1(i,:), ...
        'DisplayName', sprintf('K_d = %.1f nM', Kd_list(i)));
end
xlabel('Thoi gian (ngay)');
ylabel('%ID/g');
title('Anh huong cua ai luc (K_d) len su hap thu khang the trong khoi u');
legend('show', 'Location', 'northeast');
grid on;

%% --- Kich ban 2: anh huong cua muc bieu hien khang nguyen [Ag]/eps ---
Kd_fixed = 1;                          % [nM] ai luc co dinh (cao)
Ag_list  = [10, 100, 1000, 10000];     % [nM]
colors2  = lines(numel(Ag_list));

subplot(2,1,2); hold on; box on;
for i = 1:numel(Ag_list)
    plot(t, IDg_fun(t, Kd_fixed, Ag_list(i)), 'LineWidth', 2, ...
        'Color', colors2(i,:), ...
        'DisplayName', sprintf('[Ag]/\\epsilon = %d nM', Ag_list(i)));
end
xlabel('Thoi gian (ngay)');
ylabel('%ID/g');
title('Anh huong cua muc bieu hien khang nguyen len su hap thu');
legend('show', 'Location', 'northeast');
grid on;

sgtitle({'Mo hinh compartmental Thurber & Wittrup (2012)', ...
         'J. Theor. Biol. 314:57-68 -- nghiem giai tich Eq. (7)-(8)'});

%% --- Xuat hinh ---
exportgraphics(gcf, 'thurber_model_visualization.png', 'Resolution', 200);
fprintf('Da luu hinh: thurber_model_visualization.png\n');

%% --- (Tuy chon) Kiem tra ket qua bang so voi Table 2 cua bai bao ---
% Voi Kd rat nho (ai luc rat cao) va Ag_eps lon, uptake toi da xap xi
% cong thuc Eq. 18 trong bai bao cho truong hop ai luc cao:
tmax_idx = find(IDg_fun(t, 0.1, 10000) == max(IDg_fun(t, 0.1, 10000)), 1);
fprintf('Uptake dat dinh tai t = %.2f ngay (Kd=0.1 nM, Ag/eps=10000 nM)\n', t(tmax_idx));