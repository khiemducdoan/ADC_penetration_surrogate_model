%% diffusion_1d_solve.m
%
% 1D diffusion-degradation model for ADC (antibody-drug conjugate)
% penetration into tissue:
%
%   dc/dt = D * d2c/dx2 - r * c ,   x in [0, L]
%
% Boundary conditions:
%   c(0, t)    = c0     (Dirichlet - fixed source, e.g. vessel wall)
%   dc/dx(L,t) = 0      (Neumann / zero-flux - symmetry at tissue edge)
%
% Initial condition:
%   c(x, 0) = 0
%
% Two solutions are computed and cross-validated:
%   1) Analytical  - eigenfunction (sine-series) expansion, exact for
%      homogeneous D, r.
%   2) Numerical   - implicit Crank-Nicolson finite-difference scheme
%      (unconditionally stable), general purpose.
%
% Parameters below are taken from configs/simulation/diffusion_degradation.yaml
% (mid-range values of the sampled ranges used for the surrogate-model
% training data).
%
% Outputs: PNG figures under matlab/figures/ for use in the slide deck.

clear; close all; clc;

%% --- Parameters (mid-range of configs/simulation/diffusion_degradation.yaml) ---
D    = 10.0;        % um^2/s   - diffusion coefficient (range [1, 50])
r    = 1.0e-4;       % 1/s      - degradation / loss rate (range [1e-6, 1e-3])
c0   = 1.0;          % nM       - fixed boundary concentration (range [0.01, 10])
L    = 150.0;        % um       - tissue penetration depth (domain length)
nx   = 100;          % spatial grid points
n_modes = 150;       % eigenfunction modes for analytical series
t_max = 259200.0;    % s, 3 days

lambda_char = sqrt(D / r);  % characteristic penetration depth of steady state

fprintf('Parameters: D=%.3g um^2/s, r=%.3g 1/s, c0=%.3g, L=%.3g um\n', D, r, c0, L);
fprintf('Characteristic penetration depth lambda = sqrt(D/r) = %.3f um\n', lambda_char);

%% --- Output folder for figures ---
figDir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

%% --- Spatial grid ---
x = linspace(0, L, nx)';

%% --- Snapshot times spanning the diffusive timescale up to t_max ---
t_snap = [0, 500, 2000, 8000, 30000, 100000, t_max];

%% --- Analytical solution: steady state + transient at snapshot times ---
c_ss = steady_state_profile(x, D, r, c0, L);

c_ana = zeros(nx, numel(t_snap));
for k = 1:numel(t_snap)
    c_ana(:, k) = analytical_transient_profile(x, t_snap(k), D, r, c0, L, n_modes);
end

%% --- Numerical solution: Crank-Nicolson FDM at the same snapshot times ---
c_fdm = fdm_crank_nicolson(D, r, c0, L, nx, t_snap);

%% --- Relative error between analytical and FDM ---
rel_err = zeros(1, numel(t_snap));
for k = 1:numel(t_snap)
    denom = max(max(abs(c_ana(:, k))), 1e-12);
    rel_err(k) = max(abs(c_ana(:, k) - c_fdm(:, k))) / denom;
end
fprintf('\n%10s %18s\n', 't (s)', 'max rel. error');
for k = 1:numel(t_snap)
    fprintf('%10.1f %18.3e\n', t_snap(k), rel_err(k));
end

%% ================= Figure 1: concentration profiles =================
fig1 = figure('Color', 'w', 'Position', [100 100 900 600]);
hold on; box on;
cmap = turbo(numel(t_snap));
for k = 1:numel(t_snap)
    plot(x, c_ana(:, k), '-', 'Color', cmap(k, :), 'LineWidth', 2);
end
for k = 1:numel(t_snap)
    plot(x(1:5:end), c_fdm(1:5:end, k), 'o', 'Color', cmap(k, :), ...
        'MarkerFaceColor', 'w', 'MarkerSize', 5, 'LineWidth', 1.2);
end
xlabel('Depth x (\mum)'); ylabel('Concentration c (nM)');
title('1D Diffusion-Degradation: Concentration Profiles (line = analytical, o = FDM)');
legendStr = arrayfun(@(t) sprintf('t = %.0f s', t), t_snap, 'UniformOutput', false);
legend(legendStr, 'Location', 'northeast');
set(gca, 'FontSize', 12);
exportgraphics(fig1, fullfile(figDir, 'fig1_profiles.png'), 'Resolution', 200);

%% ================= Figure 2: space-time heatmap (log-time axis) =================
% Diffusive equilibration timescale ~ L^2/D is short compared to t_max, so a
% linear time axis squashes all the interesting transient into a sliver near
% t=0. Use log-spaced time samples (plus t=0) so the diffusion front is
% visible, then plot against log10(t) with real tick labels in seconds.
t_dense = [0, logspace(0, log10(t_max), 200)];
c_dense = analytical_transient_profile(x, t_dense, D, r, c0, L, n_modes);

fig2 = figure('Color', 'w', 'Position', [100 100 900 600]);
imagesc(log10(t_dense(2:end)), x, c_dense(:, 2:end));
set(gca, 'YDir', 'normal');
colormap(turbo); cb = colorbar; cb.Label.String = 'Concentration c (nM)';
xlabel('Time t (s)  [log scale]'); ylabel('Depth x (\mum)');
title('Space-Time Evolution of Concentration (analytical)');
tickvals = [1, 10, 1e2, 1e3, 1e4, 1e5, t_max];
set(gca, 'XTick', log10(tickvals), 'XTickLabel', arrayfun(@(v) sprintf('%g', v), tickvals, 'UniformOutput', false));
set(gca, 'FontSize', 12);
exportgraphics(fig2, fullfile(figDir, 'fig2_spacetime_heatmap.png'), 'Resolution', 200);

%% ================= Figure 2b: 3D surface c(x,t) =================
% Same data as the heatmap, rendered as a 3D surface so the build-up of the
% concentration front is visible as relief rather than just color.
[T3, X3] = meshgrid(log10(t_dense(2:end)), x);
Z3 = c_dense(:, 2:end);

fig2b = figure('Color', 'w', 'Position', [100 100 950 700]);
surf(T3, X3, Z3, 'EdgeColor', 'none');
shading interp; colormap(turbo);
cb = colorbar; cb.Label.String = 'Concentration c (nM)';
xlabel('Time t (s)  [log scale]'); ylabel('Depth x (\mum)'); zlabel('Concentration c (nM)');
title('3D Surface: Concentration c(x,t) (analytical)');
tickvals = [1, 10, 1e2, 1e3, 1e4, 1e5, t_max];
set(gca, 'XTick', log10(tickvals), 'XTickLabel', arrayfun(@(v) sprintf('%g', v), tickvals, 'UniformOutput', false));
view(-35, 28);
camlight('headlight'); lighting gouraud; material dull;
set(gca, 'FontSize', 12);
exportgraphics(fig2b, fullfile(figDir, 'fig2b_surface3d.png'), 'Resolution', 200);

%% ================= Figure 3: steady-state profile =================
fig3 = figure('Color', 'w', 'Position', [100 100 900 600]);
plot(x, c_ss, 'b-', 'LineWidth', 2.5); hold on; box on;
yline(c0 / exp(1), '--r', sprintf('c_0/e'), 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
xline(lambda_char, '--k', sprintf('\\lambda = %.1f \\mum', lambda_char), 'LineWidth', 1.5);
xlabel('Depth x (\mum)'); ylabel('Steady-state concentration c_{ss} (nM)');
title('Steady-State Profile and Characteristic Penetration Depth');
set(gca, 'FontSize', 12);
exportgraphics(fig3, fullfile(figDir, 'fig3_steady_state.png'), 'Resolution', 200);

%% ================= Figure 4: analytical vs FDM validation error =================
fig4 = figure('Color', 'w', 'Position', [100 100 900 600]);
semilogy(t_snap(2:end), rel_err(2:end), 'ks-', 'LineWidth', 2, 'MarkerFaceColor', 'k');
xlabel('Time t (s)'); ylabel('Max relative error |analytical - FDM| / max(analytical)');
title('Cross-Validation: Analytical (series) vs. Crank-Nicolson FDM');
grid on; set(gca, 'FontSize', 12);
exportgraphics(fig4, fullfile(figDir, 'fig4_validation_error.png'), 'Resolution', 200);

%% ================= Figure 5: penetration depth over time =================
% Define penetration depth as the x where c(x,t) = c0/e (63% attenuation)
target = c0 / exp(1);
pen_depth = nan(size(t_dense));
for k = 1:numel(t_dense)
    profile = c_dense(:, k);
    idx = find(profile <= target, 1, 'first');
    if isempty(idx)
        pen_depth(k) = L; % source not yet attenuated to target within domain
    elseif idx == 1
        pen_depth(k) = 0;
    else
        % linear interpolation between idx-1 and idx
        x1 = x(idx - 1); x2 = x(idx);
        c1 = profile(idx - 1); c2 = profile(idx);
        pen_depth(k) = x1 + (target - c1) * (x2 - x1) / (c2 - c1);
    end
end

fig5 = figure('Color', 'w', 'Position', [100 100 900 600]);
semilogx(t_dense(2:end), pen_depth(2:end), 'm-', 'LineWidth', 2.5); hold on; box on;
yline(lambda_char, '--k', sprintf('steady-state \\lambda = %.1f \\mum', lambda_char), 'LineWidth', 1.5);
xlabel('Time t (s)  [log scale]'); ylabel('Penetration depth at c = c_0/e (\mum)');
title('Penetration Depth vs. Time');
grid on; set(gca, 'FontSize', 12);
exportgraphics(fig5, fullfile(figDir, 'fig5_penetration_depth.png'), 'Resolution', 200);

fprintf('\nAll figures saved to: %s\n', figDir);

%% ======================================================================
%%                         Local functions
%% ======================================================================

function c_ss = steady_state_profile(x, D, r, c0, L)
    % Exact steady-state solution: D*c'' - r*c = 0, c(0)=c0, c'(L)=0
    % c_ss(x) = c0 * cosh((L-x)/lambda) / cosh(L/lambda), lambda = sqrt(D/r)
    lam = sqrt(D / r);
    c_ss = c0 * cosh((L - x) / lam) / cosh(L / lam);
end

function profile = analytical_transient_profile(x, t, D, r, c0, L, n_modes)
    % Exact transient solution via eigenfunction expansion.
    % c(x,t) = c_ss(x) - w(x,t), where w solves the homogeneous-BC problem
    % with IC w(x,0) = c_ss(x). Eigenfunctions sin(k_n x),
    % k_n = (2n-1)*pi/(2L), decaying as exp(-(D k_n^2 + r) t).
    x = x(:);
    t_arr = t(:)';
    c_ss = steady_state_profile(x, D, r, c0, L);

    n = (1:n_modes)';
    k_n = (2 * n - 1) * pi / (2 * L);
    mu_n = D * k_n.^2 + r;

    xq = linspace(0, L, 2000);
    css_q = steady_state_profile(xq, D, r, c0, L);
    sin_nq = sin(k_n * xq);                 % (n_modes, 2000)
    I_n = trapz(xq, css_q .* sin_nq, 2);    % (n_modes, 1)
    B_n = (2 / L) * I_n;

    sin_nx = sin(k_n * x');                 % (n_modes, length(x))
    decay = exp(-mu_n * t_arr);             % (n_modes, length(t))
    w = sin_nx' * (B_n .* decay);           % (length(x), length(t))
    profile = c_ss - w;
end

function c_out = fdm_crank_nicolson(D, r, c0, L, nx, t_eval)
    % Implicit Crank-Nicolson finite-difference solver (unconditionally
    % stable). Dirichlet BC at x=0 (c=c0), zero-flux BC at x=L implemented
    % via a mirrored ghost node. Each requested time is reached by an
    % independent march from t=0 with its own step count.
    x = linspace(0, L, nx);
    dx = x(2) - x(1);
    t_eval = sort(t_eval(:))';
    n_unknown = nx - 1;   % unknowns are nodes i = 2 .. nx (1-indexed), i.e. i=1..nx-1 in 0-indexed
    max_steps = 3000;

    c_out = zeros(nx, numel(t_eval));

    for kk = 1:numel(t_eval)
        t_target = t_eval(kk);
        if abs(t_target) < 1e-12
            c_cur = zeros(nx, 1);
            c_cur(1) = c0;
            c_out(:, kk) = c_cur;
            continue;
        end

        if r > 0
            dt_accuracy = min(0.4 * dx^2 / D, 0.05 / r);
        else
            dt_accuracy = 0.4 * dx^2 / D;
        end
        nsteps = min(max(ceil(t_target / dt_accuracy), 1), max_steps);
        dt = t_target / nsteps;

        alpha = D * dt / (2 * dx^2);
        beta  = r * dt / 2.0;

        a_sub  = -alpha * ones(n_unknown, 1);
        b_diag = (1 + 2 * alpha + beta) * ones(n_unknown, 1);
        c_sup  = -alpha * ones(n_unknown, 1);
        a_sub(end)  = -2 * alpha;   % mirrored ghost node (zero-flux at outer boundary)
        c_sup(end)  = 0.0;

        c_cur = zeros(nx, 1);
        c_cur(1) = c0;

        for step = 1:nsteps
            % interior unknowns correspond to original indices 2..nx (1-indexed)
            inner = c_cur(2:end-1);
            d = alpha * c_cur(1:end-2) + (1 - 2*alpha - beta) * inner + alpha * c_cur(3:end);
            d_last = 2*alpha*c_cur(end-1) + (1 - 2*alpha - beta) * c_cur(end);
            d = [d; d_last];
            d(1) = d(1) + alpha * c0;

            u = thomas_solve(a_sub, b_diag, c_sup, d);
            c_cur = [c0; u];
        end

        c_out(:, kk) = c_cur;
    end
end

function out = thomas_solve(a, b, c, d)
    % Tridiagonal solver (Thomas algorithm).
    n = numel(b);
    cp = zeros(n, 1);
    dp = zeros(n, 1);
    cp(1) = c(1) / b(1);
    dp(1) = d(1) / b(1);
    for i = 2:n
        m = b(i) - a(i) * cp(i - 1);
        if i < n
            cp(i) = c(i) / m;
        else
            cp(i) = 0.0;
        end
        dp(i) = (d(i) - a(i) * dp(i - 1)) / m;
    end
    out = zeros(n, 1);
    out(n) = dp(n);
    for i = n-1:-1:1
        out(i) = dp(i) - cp(i) * out(i + 1);
    end
end
