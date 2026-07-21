%% krogh_binding_pde.m
%
% Full spatial Krogh-cylinder model for antibody transport + reversible
% antigen binding in tumor tissue (radial coordinate r, cylindrical
% symmetry), driven by a time-varying plasma antibody concentration.
%
%   d[Ab_free]/dt = D*(1/r)*d/dr( r*d[Ab_free]/dr ) ...
%                    - (kon/eps)*[Ab_free]*[Ag] + koff*[Ab_bound]
%   d[Ab_bound]/dt = (kon/eps)*[Ab_free]*[Ag] - koff*[Ab_bound] - ke*[Ab_bound]
%   d[Ag]/dt       = Rs - (kon/eps)*[Ab_free]*[Ag] + koff*[Ab_bound] - ke*[Ag]
%
% Boundary conditions:
%   -d[Ab_free]/dr |_{r=Rcap}   = (P/D)*( [Ab]_plasma(t) - [Ab_free]/eps )
%    d[Ab_free]/dr |_{r=Rkrogh} = 0
%
% This is the spatially-resolved counterpart of the compartmental
% reduction in Thurber & Wittrup (2012, J Theor Biol 314:57-68) already
% implemented in matlab/thurber_model.m. No closed-form solution exists
% for this nonlinear system (the kon*Ab_free*Ag term couples the three
% equations), so it is solved numerically with MATLAB's built-in `pdepe`
% (m=1 selects cylindrical geometry) -- no toolbox beyond base MATLAB is
% required.
%
% As a validation (in place of an analytical cross-check, which isn't
% available for a nonlinear system), the volume-averaged total antibody
% concentration from this spatial model is compared against the
% compartmental analytical prediction of thurber_model.m, using the same
% rate constants (kex derived from the same P, Rcap, Rkrogh used in the
% PDE's boundary condition). Agreement is expected in the sub-saturating
% regime (Ab_total << Ag) and once diffusion has equilibrated the radial
% profile faster than the slower plasma-clearance/internalization scales
% -- both hold for the parameters below.
%
% Parameter values are representative orders of magnitude from Thurber
% GM, Zajic SC, Wittrup KD, J Nucl Med 2007;48:995-999 and Thurber &
% Wittrup 2012 (Tables 1-2) -- adjust freely for a specific
% antibody/antigen system.

clear; close all; clc;

%% --- Parameters ---
D       = 10.0;        % um^2/s   - antibody diffusion coefficient in tissue (Table 1: IgG ~ 5-50)
D_reg   = 1.0e-6;      % um^2/s   - tiny regularizing diffusivity for Ab_bound/Ag (both effectively
                        %            immobile membrane-bound species; pdepe requires every component
                        %            with a nonzero time-derivative coefficient to also have a flux
                        %            term depending on the spatial derivative, so a pure zero-diffusion
                        %            reaction-only equation is not directly supported). Diffusion length
                        %            sqrt(D_reg*t_max) ~ 0.9 um over 10 days, negligible next to the
                        %            65 um tissue annulus and the ~0.66 um grid spacing.
Rcap    = 10.0;         % um       - capillary (vessel) radius
Rkrogh  = 75.0;         % um       - Krogh cylinder outer radius (half intercapillary distance)
eps_v   = 0.24;         % -        - interstitial (void) volume fraction
P       = 3.0e-3;       % um/s     - vascular permeability (~3e-7 cm/s)

kon     = 1.0e-3;       % 1/(nM*s) - antibody-antigen association rate (~1e6 1/(M*s))
Kd      = 1.0;          % nM       - equilibrium dissociation constant
koff    = kon * Kd;     % 1/s      - dissociation rate
ke      = 1.109 / 86400;% 1/s      - internalization/turnover rate (1.109 /day, matches thurber_model.m)

Ag0     = 100.0;        % nM       - baseline (pre-dose) effective antigen concentration, [Ag]/eps
Rs      = ke * Ag0;     % nM/s     - antigen synthesis rate that sustains Ag0 at baseline (no antibody)

% Two-phase plasma PK (same functional form as thurber_model.m, rates
% converted from 1/day to 1/s since this script works in seconds)
Ab_plasma0 = 10.0;                  % nM  - chosen << Ag0 so Ab_total << Ag (sub-saturating regime)
A_frac = 0.6;  ka = 5.00 / 86400;   % alpha phase (fast distribution)
B_frac = 0.4;  kb = 0.05 / 86400;   % beta phase  (slow clearance, t1/2 ~ 14 days)
Ab_plasma_fun = @(t) Ab_plasma0 * (A_frac * exp(-ka * t) + B_frac * exp(-kb * t));

kex = 2 * P * Rcap / Rkrogh^2;   % 1/s - extravasation rate constant (Thurber & Wittrup 2012 definition)

fprintf('Derived rates: kex = %.4e /s (%.4f /day), koff = %.4e /s, ke = %.4e /s\n', ...
    kex, kex * 86400, koff, ke);

figDir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(figDir, 'dir'); mkdir(figDir); end

%% --- pdepe setup (m=1: cylindrical coordinates) ---
m = 1;

pdefun = @(r, t, u, dudr) deal( ...
    [1; 1; 1], ...
    [D * dudr(1); D_reg * dudr(2); D_reg * dudr(3)], ...
    [ -(kon/eps_v)*u(1)*u(3) + koff*u(2); ...
       (kon/eps_v)*u(1)*u(3) - koff*u(2) - ke*u(2); ...
       Rs - (kon/eps_v)*u(1)*u(3) + koff*u(2) - ke*u(3) ] );

icfun = @(r) [0; 0; Ag0];

% Ab_bound and Ag: zero-flux at both ends (physically, neither the bound
% complex nor the antigen actually crosses the vessel wall or the outer
% symmetry boundary -- only Ab_free does, via the permeability term).
bcfun = @(rl, ul, rr, ur, t) deal( ...
    [P * Ab_plasma_fun(t) - (P/eps_v) * ul(1); 0; 0], ... % pl
    [1; 1; 1], ...                                          % ql
    [0; 0; 0], ...                                          % pr
    [1; 1; 1] );                                            % qr

nr = 100;
t_max = 10 * 86400;   % 10 days, in seconds
rmesh = linspace(Rcap, Rkrogh, nr);
tspan = unique([0, logspace(0, log10(t_max), 200)]);

sol = pdepe(m, pdefun, icfun, bcfun, rmesh, tspan);
Ab_free  = sol(:, :, 1);   % (nt x nr)
Ab_bound = sol(:, :, 2);
Ag       = sol(:, :, 3);

fprintf('Solved: %d time points x %d radial points. min/max Ab_free = [%.4g, %.4g] nM\n', ...
    numel(tspan), nr, min(Ab_free(:)), max(Ab_free(:)));

%% --- Volume-averaged total antibody vs. compartmental (Thurber) prediction ---
Ab_total_r = Ab_free + Ab_bound;                    % (nt x nr)
area_norm = (Rkrogh^2 - Rcap^2) / 2;                 % = integral of r dr over [Rcap,Rkrogh]
Ab_total_avg = zeros(numel(tspan), 1);
for k = 1:numel(tspan)
    Ab_total_avg(k) = trapz(rmesh, rmesh .* Ab_total_r(k, :)) / area_norm;
end

Omega = kex * Kd / (Ag0 + Kd) + ke * Ag0 / (Ag0 + Kd);
Ab_ratio_fun = @(t) kex * ( ...
    A_frac / (Omega - ka) * (exp(-ka*t) - exp(-Omega*t)) + ...
    B_frac / (Omega - kb) * (exp(-kb*t) - exp(-Omega*t)) );
Ab_total_compartmental = Ab_plasma0 * Ab_ratio_fun(tspan)';

rel_err_total = abs(Ab_total_avg - Ab_total_compartmental) ./ max(Ab_total_compartmental, 1e-12);
fprintf('Spatial-avg vs. compartmental Thurber model: max rel. error = %.3f, median = %.3f\n', ...
    max(rel_err_total(2:end)), median(rel_err_total(2:end)));

%% ================= Figure A: plasma PK driving curve =================
t_days = tspan / 86400;
figA = figure('Color', 'w', 'Position', [100 100 850 500]);
plot(t_days, Ab_plasma_fun(tspan), 'LineWidth', 2.5, 'Color', [0.85 0.33 0.10]);
xlabel('Time (days)'); ylabel('[Ab]_{plasma} (nM)');
title('Plasma Antibody Concentration (two-phase PK, drives the BC at r=R_{cap})');
grid on; set(gca, 'FontSize', 12);
exportgraphics(figA, fullfile(figDir, 'krogh_figA_plasma_pk.png'), 'Resolution', 200);

%% ================= Figure B/C/D: heatmaps (log-time axis) =================
tickvals = [1, 10, 1e2, 1e3, 1e4, 1e5, t_max];
species = {Ab_free, Ab_bound, Ag};
names   = {'[Ab]_{free}', '[Ab]_{bound}', '[Ag]'};
fnames  = {'krogh_figB_Ab_free_heatmap.png', 'krogh_figC_Ab_bound_heatmap.png', 'krogh_figD_Ag_heatmap.png'};

for i = 1:3
    fig = figure('Color', 'w', 'Position', [100 100 900 600]);
    imagesc(log10(tspan(2:end)), rmesh, species{i}(2:end, :)');
    set(gca, 'YDir', 'normal');
    colormap(turbo); cb = colorbar; cb.Label.String = [names{i} ' (nM)'];
    xlabel('Time t (s)  [log scale]'); ylabel('Radius r (\mum)');
    title([names{i} '(r,t) -- Krogh cylinder, R_{cap}=' num2str(Rcap) ' \mum to R_{Krogh}=' num2str(Rkrogh) ' \mum']);
    set(gca, 'XTick', log10(tickvals), 'XTickLabel', arrayfun(@(v) sprintf('%g', v), tickvals, 'UniformOutput', false));
    set(gca, 'FontSize', 12);
    exportgraphics(fig, fullfile(figDir, fnames{i}), 'Resolution', 200);
end

%% ================= Figure E: radial profiles at snapshot times =================
snap_days = [0.01, 0.1, 0.5, 1, 3, 7, 10];
snap_t = min(snap_days * 86400, tspan(end));   % clip against float round-off at t_max
snap_idx = arrayfun(@(tt) find(tspan >= tt, 1, 'first'), snap_t);
cmap = turbo(numel(snap_idx));

figE = figure('Color', 'w', 'Position', [100 100 1100 800]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile; hold on; box on;
for k = 1:numel(snap_idx)
    plot(rmesh, Ab_free(snap_idx(k), :), 'LineWidth', 2, 'Color', cmap(k, :));
end
xlabel('r (\mum)'); ylabel('[Ab]_{free} (nM)'); title('Free antibody'); grid on;
legend(arrayfun(@(d) sprintf('t=%.2gd', d), snap_days, 'UniformOutput', false), 'Location', 'best', 'FontSize', 8);

nexttile; hold on; box on;
for k = 1:numel(snap_idx)
    plot(rmesh, Ab_bound(snap_idx(k), :), 'LineWidth', 2, 'Color', cmap(k, :));
end
xlabel('r (\mum)'); ylabel('[Ab]_{bound} (nM)'); title('Bound antibody'); grid on;

nexttile; hold on; box on;
for k = 1:numel(snap_idx)
    plot(rmesh, Ag(snap_idx(k), :), 'LineWidth', 2, 'Color', cmap(k, :));
end
xlabel('r (\mum)'); ylabel('[Ag] (nM)'); title('Free antigen (depletion near vessel = binding-site barrier)'); grid on;

nexttile; hold on; box on;
for k = 1:numel(snap_idx)
    plot(rmesh, Ab_total_r(snap_idx(k), :), 'LineWidth', 2, 'Color', cmap(k, :));
end
xlabel('r (\mum)'); ylabel('[Ab]_{total} = free+bound (nM)'); title('Total antibody'); grid on;

sgtitle('Radial Profiles at Snapshot Times');
exportgraphics(figE, fullfile(figDir, 'krogh_figE_radial_profiles.png'), 'Resolution', 200);

%% ================= Figure F: 3D surface of Ab_free(r,t) =================
[T3, R3] = meshgrid(log10(tspan(2:end)), rmesh);
Z3 = Ab_free(2:end, :)';

figF = figure('Color', 'w', 'Position', [100 100 950 700]);
surf(T3, R3, Z3, 'EdgeColor', 'none');
shading interp; colormap(turbo);
cb = colorbar; cb.Label.String = '[Ab]_{free} (nM)';
xlabel('Time t (s)  [log scale]'); ylabel('Radius r (\mum)'); zlabel('[Ab]_{free} (nM)');
title('3D Surface: Free Antibody Concentration in the Krogh Cylinder');
set(gca, 'XTick', log10(tickvals), 'XTickLabel', arrayfun(@(v) sprintf('%g', v), tickvals, 'UniformOutput', false));
view(-35, 28);
camlight('headlight'); lighting gouraud; material dull;
set(gca, 'FontSize', 12);
exportgraphics(figF, fullfile(figDir, 'krogh_figF_surface3d.png'), 'Resolution', 200);

%% ================= Figure G: validation vs. compartmental model =================
figG = figure('Color', 'w', 'Position', [100 100 900 600]);
plot(t_days, Ab_total_avg, 'b-', 'LineWidth', 2.5); hold on; box on;
plot(t_days, Ab_total_compartmental, 'r--', 'LineWidth', 2);
xlabel('Time (days)'); ylabel('Volume-averaged [Ab]_{total} (nM)');
legend('Spatial PDE (this model, radially averaged)', 'Compartmental (Thurber & Wittrup 2012, Eq. 7-8)', 'Location', 'best');
title('Validation: Spatial Model Reduces to the Known Compartmental Limit');
grid on; set(gca, 'FontSize', 12);
exportgraphics(figG, fullfile(figDir, 'krogh_figG_validation.png'), 'Resolution', 200);

fprintf('\nAll figures saved to: %s\n', figDir);
