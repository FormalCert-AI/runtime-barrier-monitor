%SYSTEM2_SOS  SOS barrier certificate of System 2 (Appendix A): degree-10 B and SOS multipliers s0, s1, s2 with
%     gam - B + s0 (|x|^2 - 0.01)    SOS   (B <= gam on X0)
%     B - E[B(x+)] + s1 (|x|^2 - 1)  SOS   (E[B(x+)] <= B on S)
%     B - 1 - s2 (|x|^2 - 1)         SOS   (B >= 1 outside S)
%     B                              SOS   (B >= 0)
%  minimising gam. E[B(x+)] uses the exact moments of xi ~ U[-0.1, 0.1].
%  Needs YALMIP and MOSEK (set the two folders below). Writes ../results/synthesis/system2_sos.json.
YALMIP_DIR = '/path/to/YALMIP';
MOSEK_DIR = '/path/to/mosek/toolbox';
addpath(genpath(YALMIP_DIR)); addpath(MOSEK_DIR);

deg = 10;
yalmip('clear');
x = sdpvar(2, 1); xi = sdpvar(1, 1);
[B, cB] = polynomial(x, deg);
f = [x(1) + 0.01*(-2*x(2));  x(2) + 0.01*((0.8 + xi)*x(1) + 10*(x(1)^2 - 0.21)*x(2))];

% E[B(f(x, xi))] for xi ~ U[a, b]:  E[xi^p] = (b^(p+1) - a^(p+1)) / ((p+1)(b-a))
a = -0.1; b = 0.1;
[c, m] = coefficients(replace(B, x, f), xi);
moments = zeros(numel(m), 1);
for k = 1:numel(m)
    p = degree(m(k), xi);
    moments(k) = (b^(p+1) - a^(p+1)) / ((p+1)*(b - a));
end
EB = c.' * moments;

gS = x(1)^2 + x(2)^2 - 1;                      % S  = {gS <= 0}
gX0 = x(1)^2 + x(2)^2 - 0.01;                  % X0 = {gX0 <= 0}
gam = sdpvar(1, 1);
[s0, c0] = polynomial(x, deg); [s1, c1] = polynomial(x, deg); [s2, c2] = polynomial(x, deg);
F = [sos(s0), sos(gam - B + s0*gX0), ...
     sos(s1), sos(B - EB + s1*gS), ...
     sos(s2), sos(B - 1 - s2*gS), ...
     sos(B)];
ops = sdpsettings('solver', 'mosek', 'verbose', 0, 'sos.newton', 1, 'sos.congruence', 1);
sol = solvesos(F, gam, ops, [cB; gam; c0; c1; c2]);
assert(sol.problem == 0, sol.info);

% coefficients and exponents of B
[coef, monomials] = coefficients(replace(B, cB, value(cB)), x);
expo = zeros(numel(monomials), 2);
for i = 1:numel(monomials)
    expo(i, :) = [degree(monomials(i), x(1)), degree(monomials(i), x(2))];
end
cert = struct('degree', deg, 'lower_bound', 1 - value(gam), 'coef', double(coef(:))', 'expo', expo);
out = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'synthesis');
if ~exist(out, 'dir'), mkdir(out); end
fid = fopen(fullfile(out, 'system2_sos.json'), 'w'); fwrite(fid, jsonencode(cert)); fclose(fid);
fprintf('certified lower bound 1 - gam = %.6f\n', 1 - value(gam));
