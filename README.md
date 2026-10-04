# Sine–Jacobi numerical certificates

This package implements the numerical certificates in **“Chebyshev-Exact Acceleration under Hessian Variation, II: Computer-assisted variational lower bound.”** It constructs a dual lower bound, encloses the sine energy, and verifies a primal feasible density by contraction. The supplied fixed candidate is in `inputs/`. A complete new search, witness construction, and full verification run through one command. The package contains no saved results.

For a full experiment, follow section 6 and run `python3 -B compute.py reproduce --jobs 8 --output results`. To verify the supplied fixed candidate without rerunning optimization, use `certify` instead. An existing result is never silently substituted for a new experiment.

References below identify statements by their titles and TeX labels, so they can be located independently of page numbering. The paper supplies the analytical proofs. This README connects those proofs to the exact data, executable formulas, acceptance conditions, and output fields. No command reads, parses, or modifies the paper, and no TeX installation is required.

## 1. Mathematical problem and numerical conclusions

For a nonnegative density on the line, define

$$
E(m)=\int_{\mathbb R}m(x)^{3/2}\,dx,\qquad
K_m(x,y)=\sqrt{m(x)}e^{-|x-y|}\sqrt{m(y)},
$$

$$
\kappa(m)=\frac{2}{\lambda_0(K_m)},\qquad
H_m(s)=\log\det\left(I+\frac{s}{2}K_m\right),\qquad
k=\frac{\pi^2}{4},\qquad H_*(s)=\log\cosh\sqrt{s}.
$$

The relaxed class in “Observations and the relaxed class” (`def:classes`) consists of bounded, compactly supported densities with

$$
\int m=1,\quad \kappa(m)=k,\quad
H_m(s)=H_*(s)\quad\text{for }s\in\{32,64,\varsigma k\},\quad
\text{tr } K_m^2=\frac23.
$$

Here $\varsigma=2704.261804447028$ is an exact rational. The root-mass condition $\int\sqrt m=1$ is not imposed in the dual relaxation; the primal construction satisfies it as an additional equation.

The three numerical objects have distinct meanings:

| Object | Meaning | Principal output |
|---|---|---|
| Dual value $c$ | A lower bound for every density in the relaxed class, conditional on the analytical certificate theorem | `intervals.certificate.bound` |
| Sine energy $c_{\rm sine}=E(m_*)$ | Energy of the analytically defined sine density | `sine_recomputed.energy` |
| Feasible upper bound $U$ | An upper bound for the energy of an exactly feasible density obtained by contraction | `intervals.feasible.energy_ceiling` (upper endpoint) |

These paths are relative to `verification.json → numerical_results`.

If the verification establishes $c\in[c_-,c_+]$ and $c_{\rm sine}\in[S_-,S_+]$, then $c_-$ is a rigorous lower bound for the relaxed infimum. **The endpoint $c_+$ bounds the value of this dual certificate; it is not an upper bound for the infimum.** The feasible construction supplies that upper bound.

“Dual certificates bound the energy” (`thm:dual`), together with the analytical reduction and uniform transfer, converts $c_->0$ into the asymptotic gain lower bound $2\sqrt{c_-}$. The sine construction supplies the corresponding upper bound $2\sqrt{S_+}$. Thus the gain ratio is at most $\sqrt{S_+/c_-}$. These implications require the analytical theorems; the program does not prove the recurrence-to-density reduction, the spectral transfer, or the structural obstruction theorem.

## 2. Paper-to-code map

Paths beginning with `programs/native/` contain the numerical formulas and native acceptance predicates. `audit/` performs rational checks of their consequences. `portable/verify.py` coordinates a complete replay from an exported archive.

| Paper statement or data | Input or evidence | Code to inspect | Quantity or predicate |
|---|---|---|---|
| “Contact potential” (`thm:contact`); “The normalized criterion” (`lem:regular-criterion`), items (R)–(G) | `inputs/reference.txt`; supporting and complementary flow witnesses | `programs/native/reference.hpp`: `Reference`, `support_shape`, `gamma_ratio` | Contact profile and $\Gamma_\varsigma/Q$ |
| Calibration and target value (`def:base`, `eq:dual-target-value`); calibration integral (`lem:J-value`) | Reference flows and quadratures | `programs/native/verify_reference.cpp`; `Reference::local_shape`, `target_H`, `target_S` | `J`, `base`, `mass_1`, `mass_2`, `mass_3` |
| Signed comparisons (`thm:comparisons`); symmetric spectral formulas (`lem:symmetric-values`) | Comparison filters and linear filter | `Reference::build_filters`, `build_complements`; `verify_reference.cpp` | `H_0`, `H_1`, `H_signal_0`, `H_signal_1`, `S`, `S_signal` |
| State region (`prop:state-region`, `lem:range`) | Two conditional branches over $[0,1]^2$ | `programs/native/transport_kernel.hpp`: `geometry`, `Profile` | $(U,V,\vartheta_-,\vartheta_+)$ |
| Potential coefficients and endpoint cancellation (`eq:transport-potential`, `tab:transport-polynomials`, `eq:regular-h`, `eq:regular-u`) | `inputs/transport.txt` | `programs/native/rational.hpp`: `verify_cancellation`; `audit/checks.py`: `polynomial` | Exact polynomial identities on both spatial sides |
| Normalized pointwise criterion (`lem:regular-criterion`, `eq:witness`); cubic conjugate (`lem:conjugate`) | Reference profile, multipliers, potential, prices | `transport_kernel.hpp`: `StateFormula`, `StateKernel`; `programs/native/box_arithmetic.hpp` | Enclosures of the criterion, gradient, and Hessian |
| Box bounds (`lem:box-bound`); computed dual certificate (`ca:transport`) | `second/state-cover.bin` and reference flows | `programs/native/state_cover.hpp`: `Cover::bound`; `programs/native/verify_transport.cpp` | Complete spatial/state cover, strict leaf bounds, price integrals |
| Certified constant (`cor:numerical`, `tab:reference-enclosures`, `tab:ledger`) | Reference report and verified regional prices | `audit/verify.py`: `verify` | Signed lower-bound assembly |
| Sine energy (`lem:sine-sum`, `ca:sine`) | Sine polynomial and interval report | `programs/native/verify_sine.cpp`; `audit/exact.py`: `sine_enclosure`; `audit/checks.py`: `sine` | Integer coefficients, finite sum, infinite-tail bound |
| Primal family (`thm:primal-map`, `def:primal`) | `inputs/profile.txt`, `inputs/radius.txt` | `programs/native/primal.hpp`: `PrimalParameters`, `primal_shape` | Fixed ground parameter, six-parameter family |
| Primal derivatives (`prop:primal-jacobian`, `eq:primal-jacobian-mass`, `eq:primal-jacobian-spec`); computed data (`ca:primal-second`, `tab:primal-values`) | Central and whole-cube flows, quadrature partition | `PrimalEvaluation::integrand`, `integrate`, `residual`; `programs/native/verify_primal.cpp` | All six residuals and all 36 Jacobian entries |
| Exact feasibility (`lem:contraction`, `thm:primal`); feasible energy (`cor:primal`) | Exact preconditioner, radius, residual/Jacobian enclosures | `primal.hpp`: `report_primal`; `audit/checks.py`: `contraction` | Contraction, strict self-map, energy ceiling |
| Main energy and gain results (`thm:energy`, `thm:main-centered`) | Verified dual and sine bounds plus analytical reduction | `audit/verify.py`; manual application of the paper's theorems | Derived ratios and gaps |

`programs/stages.py` orchestrates construction. `programs/native/produce_transport.cpp` proposes prices and state covers; `programs/native/prepare_primal.cpp` builds central and cube witnesses. The `verify_*.cpp` entry points perform native acceptance. `verify_reference.cpp` and `verify_sine.cpp` also generate their respective reports. `programs/proof.py` packages evidence and invokes `portable/verify.py`; `audit/report.py` formats exact consequences, and `programs/workflow.py` writes the short summary only after full acceptance.

The finite Jacobi-matrix arguments, common-measure representation, state-region theorem, contact-potential theorem, signed comparison identities, continuity in $\kappa$, and “No differentiable sharp bound at the sine density” (`thm:no-touching`) require mathematical review of the paper. They are not replaced by a numerical test.

## 3. Exact inputs and notation

### Numerical conventions

The values and dimensions in this section describe the supplied fixed candidate. A new search can choose different calibration, support, scales, directions, coefficients and radius; its output files and verification report are authoritative. Decimal input tokens denote exact decimal rationals. A token `a/b` denotes an exact rational quotient. A hexadecimal token such as `0x3p-2` denotes an exact dyadic rational, here $3/4$. Interval records contain two exact endpoints in lower–upper order. Decimal output is a presentation of these endpoints, rounded outwards; it should not be converted through binary `float` when checking inequalities.

| Code name | Mathematical meaning |
|---|---|
| `beta` | Calibration parameter $\varsigma$ |
| `gamma` in the parameter file | Calibration multiplier $\omega$; distinct from the function $\gamma(t)=\Gamma_\varsigma/Q$ |
| `k` | $\pi^2/4$, enclosed at the requested precision |
| `support` | $(p_0,p_1,\zeta_1,\ldots,p_5,\zeta_5)$ |
| `scales` | $(s_0,s_1)=(32,64)$ |
| `multipliers` | $(\xi_0,\xi_1,\tau,\chi)$, in this order |
| `reference.J` | Calibration integral $\mathcal J_\varsigma(2)$ |
| `reference.mass_j` | $\int\bar m_k^{j/2}$, for $j=1,2,3$ |
| `reference.H_signal_i` | $H_*(s_i)-H_{\bar m_k}(s_i)$ |
| `reference.S` | $S(\bar m_k)=\frac{k^2}{4}\text{tr }K_{\bar m_k}^2$ |
| `reference.S_signal` | $k^2/6-S(\bar m_k)$ |
| `feasible.F_i`, `feasible.J_i_j` | Zero-based residual and Jacobian indices; paper indices are one-based |
| Sine truncation `J` in JSON | The paper's truncation index $K$; unrelated to the calibration integral or Jacobian |
| `second`, `primal_second` | Dual-certificate and feasible-density stage directories |

### Reference data

`inputs/reference.txt` has header `CONTACT_REFERENCE 1`, followed by $\varsigma,b,\omega$, a count and the support vector, then a count and the scales. It defines

$$
b_\circ(Q)=p_0+\sum_{j=1}^{5}p_j\frac{Q}{Q+\zeta_j},\qquad
b=0.11085839213923356,\qquad \omega=1351.3424903216799.
$$

The support vector corresponds to `tab:cert-data`; the three scalars correspond to `eq:cert-scalars`. Positive coefficients give the positivity and monotonicity required by the contact-potential theorem. The supporting density $m_\circ$, the comparison density $\bar m_k$, and the feasible density $m_{a_*}$ are different objects.

### Transport data

`inputs/transport.txt` has header `TRANSPORT_POLYNOMIAL 1`, the number of scales and their values, four multipliers in the order above, and a term count. Each term contains its kind (`h` or `u`), powers $(i,j)$, an exact coefficient, and four polynomials for each of the two spatial sides. Every polynomial is stored as its coefficient count followed by coefficients in ascending powers of $q$.

The six terms represent

$$
\Psi_{\rm raw}=(2-q)^2\left[P_hh+qP_0u_0+qP_1u_1
+P_{00}u_0^2+P_{01}u_0u_1+P_{11}u_1^2\right].
$$

To compare with `tab:transport-polynomials`, restore the endpoint factors shown in this formula. The serialized data are the endpoint-cancelled polynomials used in evaluation, not simply the six four-coefficient rows of that table.

For a full coefficient polynomial $P$ multiplying $u_0^iu_1^j$, let $d=i+j$. On side $s$, the four stored polynomials are

$$
\mathscr D_s(P'q^d),\quad \mathscr D_s(Pq^{d-1}),\quad
\mathscr D_s(Pq^d),\quad \mathscr D_s(Pq^{d+1}),
$$

where $\mathscr D_{\rm L}f=f/q$ and $\mathscr D_{\rm R}f=f/(2-q)$. For a term $Ph$, they are $\mathscr D_s(P'q),\mathscr D_s(Pq),0,0$. Both C++ and Python reconstruct and check these identities over the rationals; evaluation near an endpoint does not divide by an interval containing zero.

### Primal data

`inputs/profile.txt` has header `GROUND_PRIMAL 1`, then $\varsigma$, the second-trace flag, the scale count and scales, the coefficient count and coefficients, the correction dimension and direction indices, and the preconditioner in row-major order. There are 17 exact dyadic coefficients, six directions $(0,1,2,3,6,7)$, and a $6\times6$ exact dyadic matrix. Compare these with `tab:primal-second-coefficients`, `def:primal`, and `tab:primal-second-matrix`.

`inputs/radius.txt` contains the point interval $[\rho,\rho]$, where

$$
\rho=27684846429260304880501991746477461659874142439\,2^{-207}.
$$

The fixed coefficients and preconditioner are not changed by `prepare_primal`. `certify` constructs witnesses for these inputs. `reproduce` first generates a new set in its own output directory (section 7). Neither command modifies the supplied `inputs/`.

### Work configuration

`inputs/run.json` controls construction and replay:

| Stage | MPFR bits | Taylor order | Flow tolerance | Quadrature tolerance |
|---|---:|---:|---:|---:|
| `reference` | 128 | 12 | `1e-20` | `1e-14` |
| `second` | 128 | 12 | `1e-20` | `1e-14` |
| `primal_second` | 160 | 14 | `1e-24` | `1e-18` |

The sine calculation uses 160 bits and an absolute tail target of `1e-18`. Transport construction uses `price_error = 2^-21`, `state_budget = 4096`, 16 initial spatial segments per half, and a target of eight verification regions. These are work parameters, not substitutes for the acceptance inequalities. Changing them can change partitions, prices, interval widths, and the attained certificate value.

## 4. Mathematical acceptance conditions

### 4.1 Reference flows and integrals

Set $Q(q)=q(2-q)$ and use $q=t^3$ on the left side and $q=2-t^3$ on the right, with $0\le t\le1$. A full flow uses the increasing coordinate $v\in[0,2]$, with $t=v$ on the left and $t=2-v$ on the right. The comparison amplitude is $\nu_k=t\hat\nu$, where

$$
k\nu_k^3-3Q\nu_k-2bQ-2k\omega\Gamma_\varsigma=0.
$$

The supporting amplitude uses $b_\circ(Q)$ in place of $b+k\omega\Gamma_\varsigma/Q$. The contact identity defines $\Gamma_\varsigma$ from the supporting filter and its reflection. `Reference` implements items (R), (G), (C), and (F) preceding `lem:regular-criterion`.

The comparison Riccati and linear filters satisfy

$$
r_s'=s\bar m_k-2r_s-r_s^2,\qquad
r_\circ'=k\bar m_k-2r_\circ,
$$

with zero incoming values. Complementary flows evaluate $q-kr_s/s$ and $q-r_\circ$ without cancellation at $t=0$; `CubicQuotient` encloses their quotients by $t^3$ using the removable-zero identity, including the endpoint.

For a symmetric comparison density, the exact reflection formulas are

$$
H_{\bar m_k}(s)=2\int_{\text{left half}}r_s\,dx+\log(1+r_s(\text{midpoint})),
$$

$$
S(\bar m_k)=2\int_{\text{left half}}r_\circ^2\,dx+r_\circ(\text{midpoint})^2.
$$

These terms include the exterior contributions analytically; there is no numerical truncation of the infinite line. The base value is

$$
\mathcal B_{*}=-b+\frac{2b}{k}+\frac{3}{2k}\int_0^2\nu_k\,dq
-\omega\mathcal J_\varsigma(2)
-\omega\left(\frac{k}{\varsigma}-\frac{2H_{*}(\varsigma k)}{\varsigma^2}\right).
$$

`verify_reference.cpp` outputs its enclosure and the comparison masses and signals. In particular, `mass_3` is the energy of the comparison density, whereas `base` is the calibration lower-bound term.

### 4.2 State inequality and exhaustive coverage

At $\kappa=k$, use the normalized state coordinates from `lem:range`, with $U_0=U$ and

$$
U_1=V(U,T)=(1-T)\frac{(k+s_0)U}{k+s_1-(s_1-s_0)U}+TU.
$$

The remaining moment coordinate ranges over $[\vartheta_-(U,T),\vartheta_+(U,T)]$. The normalized criterion is convex in this coordinate, so its two endpoints suffice. The function bounded on every spatial cell $\mathsf C$ and state rectangle is

$$
\Phi_\sigma(t,U,T)=\psi(\mathfrak b_\sigma-e(t))-\mathfrak c_\sigma-e(t)(1-\mathfrak l),
\qquad
\psi(x)=-x-\frac12+\frac4{27}(x+\tfrac32)_+^3.
$$

`StateKernel` evaluates the coefficients in the regrouped form of `lem:regular-criterion`. `Cover::bound` combines interval evaluation, monotonicity, gradient/Hessian Taylor bounds, and a concavity bound when its hypotheses are verified (`lem:box-bound`). The midpoint, gradient, and Hessian estimates cover the entire spatial cell. Sampled values may guide construction but cannot establish acceptance.

For each side and cell, the verifier requires both conditional branches to form complete binary subdivision trees of $[0,1]^2$. Spatial cells must form an ordered partition of each complete half. At every leaf it recomputes an upper bound and requires

$$
\text{recomputed upper bound}\le\text{recorded upper bound}<0.
$$

Thus acceptance establishes a continuous inequality on the whole domain, including endpoints, rather than a finite grid test. The price integral is independently recomputed and must be contained in its recorded interval.

### 4.3 Common dilation and signed lower-bound assembly

The cover can use a cell dilation $0<\lambda_{\mathsf C}\le1$. This scales the storage potential and all four multipliers; the stored affine price $e_{\mathsf C}$ is the price accepted at that scale. The final certificate uses a common $0<\lambda\le\min_{\mathsf C}\lambda_{\mathsf C}$ and the cell price $(\lambda/\lambda_{\mathsf C})e_{\mathsf C}$.

This reduction is valid because $\psi$ is convex and $\psi(0)=0$. For $\alpha=\lambda/\lambda_{\mathsf C}\in(0,1]$,

$$
\psi(\alpha x)-\alpha y\le\alpha\bigl(\psi(x)-y\bigr).
$$

Consequently every strict accepted cell inequality remains strict at the common scale. This argument must accompany the use of the `dilation` field; prices from different scales cannot simply be added unchanged.

Let $I_{\mathsf C}$ enclose the integral of the stored cell price with weight $3t^3\hat\nu/k$. Then

$$
\Pi=\lambda\sum_{\mathsf C}\frac{I_{\mathsf C}}{\lambda_{\mathsf C}},\qquad
c=\mathcal B_*+\lambda\left(\xi_0\delta H_0+\xi_1\delta H_1
+\chi\delta S-\tau(1-\bar M)\right)-\Pi.
$$

`undilated_price` is the sum before multiplication by the common $\lambda$; `price` already includes that factor. `programs/price_claims.py` assembles provisional regional intervals from the stored cell integrals. It widens their exact rational sums by a conservative directed-rounding allowance, using absolute magnitudes to cover cancellation. These are claims, not verified results. During full verification, each native regional reader recomputes every selected cell integral and requires both its regional price intervals to lie inside these claims. `audit.checks.merge_prices` requires exact regional coverage and agreement of the sums, dilation, and counts; `audit.verify.verify` supplies the signed lower-bound formula. In particular, the mass contribution is $-\tau(1-\bar M)$, and the price is subtracted once.

The resulting certificate agrees with the paper's unscaled potential and multipliers only when $\lambda=1$. A different partition or price can also define a different certificate even with identical fixed inputs. Its validity and its agreement with the paper's printed value are separate questions.

### 4.4 Primal feasibility on an entire cube

Define

$$
\phi_a(t)=\sum_{j=0}^{16}c_jT_j(2t-1)+\sum_{i=0}^{5}a_iT_{d_i}(2t-1),
\quad(d_0,\ldots,d_5)=(0,1,2,3,6,7),
$$

$$
D_a(t^3)=t^2e^{\phi_a(t)},\qquad D_a(2-q)=D_a(q),\qquad
m_a(X_a(q))=\frac{D_a(q)+Q(q)}{k},\quad X_a'(q)=D_a(q)^{-1}.
$$

The prescribed-ground-parameter theorem establishes $\kappa(m_a)=k$. In code, the left-half derivative with respect to $t$ is $dx/dt=3e^{-\phi_a(t)}$.

The zero-based residual order is

$$
F(a)=\left(\int m_a-1,\ \int\sqrt{m_a}-1,\
H_{m_a}(32)-H_{*}(32),\ H_{m_a}(64)-H_{*}(64),\
H_{m_a}(\varsigma k)-H_{*}(\varsigma k),\ S(m_a)-k^2/6\right).
$$

The verifier replays the central flows and flows with every selected coefficient replaced by $c_{d_i}+[-\rho,\rho]$. The latter enclose every parameter vector in the cube simultaneously. They are used with the analytic derivative formulas to enclose all entries of $DF([-\rho,\rho]^6)$; evaluating the Jacobian only at the center would not suffice.

From the exact preconditioner $C$, the rational audit recomputes

$$
\eta=\max_i\text{mag }\left(\sum_jC_{ij}[F_j(0)]\right),\qquad
\theta=\max_i\sum_j\text{mag }\left(\delta_{ij}-\sum_\ell C_{i\ell}[\partial_jF_\ell]\right).
$$

It requires $\theta<1$ and $\eta+\theta\rho<\rho$, and also verifies that the separately reported norm bounds dominate these recomputed bounds and establish the same strict self-map. The bound $\theta<1$ also implies that the square preconditioner $C$ is nonsingular. The contraction theorem applied to $a\mapsto a-CF(a)$ therefore gives a unique zero of $F$ in the cube. The energy theorem gives

$$
E(m_{a_*})\le E(m_0)e^{6\rho}\le U.
$$

The exponential bound is reevaluated with rational enclosures. The program reports `strict_feasible_improvement = true` precisely when the independently enclosed sine lower endpoint exceeds this recomputed feasible energy ceiling.

### 4.5 Sine sum and infinite tail

For truncation index $K$, form the integer polynomial

$$
\left(\sum_{j=0}^{K}a^{j(j+1)}\right)^6
\left(1+2\sum_{j=1}^{K}(-1)^ja^{j^2}\right)^6=\sum_nC_na^n.
$$

With

$$
I(z)=e^{-\pi z}\left(\frac1{\pi z}+\frac2{(\pi z)^2}+\frac2{(\pi z)^3}\right),\qquad
E_K=512\sum_n C_nI(n+3/2),
$$

`lem:sine-sum` supplies $|c_{\rm sine}-E_K|\le R_K$, where

$$
R_K=512\left[
\frac{6M_1^5M_2^6I(3/2+(K+1)(K+2))}{1-e^{-2(K+2)\pi}}
+\frac{12M_1^6M_2^5I(3/2+(K+1)^2)}{1-e^{-(2K+3)\pi}}
\right],
$$

$$
M_1=(1-e^{-2\pi})^{-1},\qquad M_2=1+\frac{2e^{-\pi}}{1-e^{-3\pi}}.
$$

The native implementation uses the equivalent substitution $u=\pi\varpi$: its elementary integral is $\int_\pi^\infty u^2e^{-zu}\,du$ and its prefactor is $512/\pi^3$. The supplied settings selected $K=3$ (degree 126) in the recorded calculation discussed in the paper; a new report records the actual index.

The Python checker reconstructs every integer coefficient and reevaluates the finite sum and tail using rational enclosures. It obtains $\pi$ from Machin's arctangent identity with alternating-series bounds, and exponentials from Taylor bounds and range reduction. It requires the saved energy and finite-sum intervals to contain these enclosures and the saved tail upper bound to dominate the rational tail bound.

## 5. Enclosure rules and trusted components

| Mathematical rule | Implementation | What a reviewer should check |
|---|---|---|
| Directed scalar interval arithmetic | `programs/native/interval.hpp`, `mpfr_api.hpp` | Correct lower/upper MPFR rounding, domain checks, rational input interpretation |
| Exact polynomial algebra | `programs/native/rational.hpp`; `audit/exact.py`, `audit/checks.py` | Multiplication, differentiation, endpoint factors and coefficient order |
| Normalized derivatives $f^{(j)}/j!$ (`lem:jets`) | `programs/native/series.hpp`; `flow.hpp`: `ode_jet` | Product, inverse, exponential, square-root, cubic-root and Riccati recurrences |
| Positive cubic root (`lem:cubic-root`) | `interval.hpp`, `series.hpp` | Positive initial bracket and every interval-Newton narrowing |
| Validated flow (`lem:flow-step`) | `flow.hpp`: `Flow::read_saved`, `load_and_verify` | Incoming enclosure, exact Taylor center, Picard tube, coefficient containment, remainder and one-sided derivative bound |
| Removable cubic zero (`lem:cubic-quotient`) | `flow.hpp`: `CubicQuotient` | Taylor/integral enclosure of $f(t)/t^3$, including $t=0$ |
| Taylor quadrature (`lem:quadrature`) | `programs/native/quadrature.hpp`; `PrimalEvaluation::cell_integral` | Integral of the central polynomial plus a bound using derivatives on the entire cell |
| Binary64 box arithmetic and polynomial translation (`lem:translation`) | `programs/native/box_arithmetic.hpp` | Outward widening of operations, derivative rules, translated polynomial evaluation |
| Box maximization (`lem:box-bound`) | `programs/native/state_cover.hpp`: `Cover::bound` | Hypotheses and direction of every upper-bound and monotonicity operation |
| Exact consequences | `audit/exact.py`, `audit/checks.py`, `audit/verify.py` | Signed interval algebra, six-dimensional norms, ratios, and gaps |

For $y'=C+Ay+By^2$, a flow witness contains an exact Taylor center $m$, a radius $r$ enclosing the incoming state about $m$, a Picard tube $Y$, a polynomial of order $N$, a bound $R$ on the normalized derivative of order $N+1$, and a one-sided derivative bound $L$. Replay checks the Picard image for the hull of the incoming interval and $m$, then validates

$$
y(a+v)\in\left[\sum_{j=0}^N z_jv^j-Rv^{N+1}-re^{Lv},\
\sum_{j=0}^N z_jv^j+Rv^{N+1}+re^{Lv}\right]\cap Y.
$$

A negative $L$ is retained. Saved coefficients, tubes, and terminal intervals are acceptance targets, not assumed ODE solutions. `load_proposal` is used only to construct candidates; acceptance uses `load_and_verify`.

For quadrature on $[c-h,c+h]$, the remainder bound is $2h^{N+1}\sup|f_N|/(N+1)$ after integrating the even central Taylor terms. Reflection uses the appropriate one-sided jets at the shared midpoint.

MPFR intervals use directed rounding. State boxes use IEEE binary64 round-to-nearest followed by widening to neighboring representable values. Runtime checks require binary64 representation, round-to-nearest, and gradual underflow. Compiler flags include `-ffp-contract=off -fno-fast-math`; changing those flags changes the arithmetic assumptions. The rational audit limits integer growth by outward dyadic rounding at 300 bits where specified. Decimal square-root bounds use integer square roots and outward rounding.

The trust basis comprises the analytical lemmas, their implementation, Python and the C++ compiler/runtime, GMP/MPFR, and the checked floating-point environment. The standalone verifier runs independently of the producer process but shares its native mathematical kernels. It is not a second implementation of those kernels or a formal proof assistant. The rational sine evaluation, exact polynomial checks, and separate Python cover scanner provide additional checks of their respective components.

## 6. Installation and the reproduction protocol

Use macOS or Linux, Python 3.10 or newer, a C++17 compiler, and GMP/MPFR development files for fixed-input construction and verification. These modes use only Python's standard library and the native dependencies. For the complete search, use Python 3.13 (used for the supplied candidate) with the pinned NumPy 2.3.5 and SciPy 1.17.0 dependencies.

On macOS, install the command-line compiler and libraries:

```sh
xcode-select --install
brew install gmp mpfr
export SINEJACOBI_NATIVE_PREFIX="$(brew --prefix)"
```

`SINEJACOBI_NATIVE_PREFIX` must contain `include/` and the static archives `lib/libmpfr.a` and `lib/libgmp.a`. With a standard system installation on Linux, leave it unset. For example, on Debian/Ubuntu:

```sh
sudo apt-get install build-essential libgmp-dev libmpfr-dev python3-venv
```

From the extracted package directory, prepare the search environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r programs_search/requirements.txt
```

The following optional probe compiles and runs a small arithmetic/environment check; it performs no search or certificate calculation:

```sh
python -B compute.py doctor
```

The complete protocol is one command:

```sh
python -B compute.py reproduce --jobs 8 --output results
```

This executes search → interval-witness construction → archive packaging → full standalone verification. The default settings are `programs_search/settings.json`; pass `--settings PATH` to use another file. `--jobs` bounds concurrent numerical processes, not the number of mathematical regions. The launcher sets BLAS/OpenMP thread limits to one; the worker count and environment are recorded. Use `CXX` to select a compiler executable. Keep the strict floating-point compilation flags unchanged.

For the supplied exact candidate, bypass search:

```sh
python -B compute.py certify --inputs inputs --jobs 8 --output results-fixed
```

This still constructs all witnesses and performs the same full verification. It needs no NumPy/SciPy. To use previously generated candidates, pass their directory as `--inputs`.

For a new replay of an existing archive:

```sh
python -B compute.py verify --proof results/proof.tar.xz --jobs 8 --output verification-new.json
```

`verify-proof` is an alias of `verify`, retained for the article's earlier command spelling. Both call the same verifier. Replaying an archive neither runs the search nor rebuilds the proposed certificate. A new `reproduce` or `certify` run requires an absent or empty output directory.

Resume a failed/interrupted run explicitly with the original command and `--resume`:

```sh
python -B compute.py reproduce --jobs 8 --output results --resume
```

Resume requires unchanged sources, settings and selected exact inputs. A completed candidate snapshot is reused; otherwise the search restarts. Completed construction stages are reused only when their source/input/toolchain receipts and output hashes match. A failed incomplete construction stage is rebuilt. If packaging already completed, the bound archive is reverified. A completed experiment is replayed with `verify`, or repeated into another output directory. Resume never accepts an unrelated archive or silently overwrites old data.

Successful runs remove `work/` by default. `--keep-work` retains search history, construction logs, native build receipts and intermediate witnesses, as well as the verifier's temporary directory (its path is recorded in `verification.json`). Failures retain diagnostics automatically. Per-evaluation search dumps are off by default; `SINEJACOBI_SEARCH_TRACE_MODE=full` enables them for diagnosis and should be combined with `--keep-work`.

## 7. What the generator computes

The generator does not read the fixed candidate or the article. `programs_search/settings.json` contains budgets and tolerances, with no calibration, profile, multiplier, matrix or radius answer.

| Step | Computation | Source in `programs_search/` |
|---|---|---|
| Contact stationarity | Solve mass normalization, continue the stationary boundary-value problem over calibration scales, refine the best computed energy | `search_reference.py` |
| Positive rational support | Fit positive poles/amplitudes in logarithmic coordinates; refine comparison multipliers | `fit_reference.py`, `model.py` |
| Observation scales | Compare pairs of powers of two, refine promising neighborhoods, reevaluate finalists | `search_transport.py` |
| Polynomial potential | Optimize on coarse then fine spatial/state grids, first without and then with the second-trace observation | `search_transport.py`, `certificate_format.py` |
| Primal density | Fit log drift by a Chebyshev polynomial and numerically minimize energy, first with five constraints and then with all six | `search_primal.py`, `primal_model.py` |
| Correction directions | Choose Jacobian columns by pivoted QR and compute an approximate inverse | `format_primal.py` |
| Exact primal candidate | Convert center/preconditioner to dyadics, correct the center, choose a cube radius from its residual | `native/produce_primal.cpp` |

The support is $p_0+\sum_j p_jQ/(Q+\zeta_j)$, $Q=q(2-q)$, with positive coefficients and poles. Its number of terms and coefficients come from the fit. Calibration continuation starts at unit scale and doubles it before refining the successful range. The primal family uses $q=t^3$, $D=t^2e^{\phi(t)}$, $dx/dt=3e^{-\phi(t)}$, and $m=(D+Q)/k$. The polynomial coefficients are optimization variables. Correction directions are selected from the computed Jacobian; the indices in section 4.4 describe the supplied candidate, not hard-coded search directions.

After interval evaluation of the center, let $\eta$ be the upper bound on the correction norm. The preliminary native search chooses

$$
\rho=\text{round }_{\uparrow}(2\eta+2^{-p}),
$$

where $p$ is the primal precision (160 bits by default), once $2\eta$ is smaller than `radius_goal`. It then evaluates the whole cube. This correction serves candidate selection; subsequent construction and the final reader reconstruct/replay the witnesses for the fixed selected candidate. A numerically optimized candidate is not claimed to be a global minimizer.

The proposed transport price tolerance is the power of two at or below `sampled_increment / price_accuracy_factor`. Radius goals, grids, optimization tolerances and price work targets guide construction; they replace no acceptance inequality.

Defaults: 11 stationary-search digits, 12 fit digits, at most 10 positive-support terms, 9 initial scale levels, state degree 2; scale-search `(order, grid, degree, iterations) = (32,17,3,300)`, transport coarse `(64,33,3,600)`, transport fine `(80,41,4,800)`; primal degree 16, quadrature order 192, iteration budget 800, radius goal `1e-12`; price accuracy factor 32. The `verification` entries supply the native settings in section 3.

To stop after candidate generation:

```sh
python -B compute.py generate --jobs 8 --output generated-inputs --work search-work
python -B compute.py certify --inputs generated-inputs --jobs 8 --output results-generated
```

These use the same generator and certification workflow as `reproduce`. Both generator directories must be absent or empty. `CANDIDATES_GENERATED` is only a search status. Optimization and platform differences can produce different candidates; even identical fixed inputs can lead to different partitions and interval widths. Full acceptance establishes the new certificate's values, without comparing them to previously published digits.

## 8. One acceptance protocol and distinct verifier roles

| Component | Role |
|---|---|
| `programs/stages.py` | Internal witness construction; never issues `PASS_FULL_NUMERICS` |
| `programs/proof.py` | Export and isolated launcher; packaging status is `EXPORTED_NOT_VERIFIED` |
| `portable/verify.py` | Sole complete acceptance coordinator, used by all full-run/replay commands |
| `verify_proof.py` | Thin launcher; copied as `verify.py` into a standalone proof |
| `native/verify_reference.cpp` | Validate the supporting flow and recompute reference flows, quadratures, masses and signals |
| `native/verify_primal.cpp` | Replay central/whole-cube ODE witnesses, recompute residual/Jacobian and contraction |
| `native/verify_transport.cpp` | Validate prerequisite flows, every assigned state leaf and price integral; check spatial coverage |
| `native/verify_sine.cpp` | Reconstruct integer polynomial, finite sum and infinite-tail enclosure |
| `portable/cover.py` | Packed-tree geometry, complete spatial coverage, counts and exact cell-price bookkeeping |
| `audit/checks.py`, `audit/verify.py` | Exact endpoint identities, regional union, signed bound, contraction and energy consequences; rational sine cross-check |
| `tests/`, `programs_search/test_search.py` | Optional developer regression tests, outside certificate acceptance |

`native/` here means `programs/native/` in the source package and `native/` in the archive. There is one shared set of native headers. Search retains its distinct `produce_primal.cpp` entry point and includes those headers.

The complete reader validates identities and geometry, checks rational consistency, recompiles the bundled verifier, rechecks reference/primal/sine tasks, then verifies all transport regions. It returns `PASS_FULL_NUMERICS` only after every task and containment test succeeds. Conditional exact-audit status is internal; it is never presented as a complete certificate. There are no public partial-audit, smoke-acceptance or publish commands.

Construction no longer repeats native primal/transport acceptance immediately before the standalone replay. Reference and sine construction still need their numerical routines to produce initial claims; the final reader recomputes them. Each transport worker revalidates its prerequisite flows because its region must not depend on an unchecked cache; this bounded prerequisite repetition is intentional. Python polynomial/sine/contraction checks cross-check different arithmetic layers, rather than creating competing acceptance protocols.

No threshold such as a number of matching published digits is coded into acceptance. `PASS_FULL_NUMERICS` may describe a weaker certificate. `strict_feasible_improvement` separately records whether the verified upper bound beats the sine lower bound.

Developer checks, outside the reproduction protocol:

```sh
python -B compute.py test
python -B compute.py test --native --search
```

The first runs standard-library infrastructure/exact-arithmetic tests. The second also compiles and runs small native regressions and requires the search dependencies. Synthetic fixtures cannot be accepted as production proofs. Tests do not replace the complete experiment.

## 9. Results to retain and return

A successful output directory contains:

| File | Purpose |
|---|---|
| `proof.tar.xz` | Exact inputs, interval claims, ODE/Taylor witnesses, packed state cover, settings and verifier sources |
| `verification.json` | Full acceptance status; exact/outward-decimal results; coefficients, residuals, Jacobian; geometry; actual compiler/GMP/MPFR versions; requested workers and verifier timings |
| `run.json` | Mode, source/input hashes, settings, complete generation summary, host information, phase statuses and wall times |
| `results.txt` | Short outward-rounded summary, written only after full acceptance |
| `inputs/` | Five exact inputs actually used, with search provenance when available |

A small lock file is retained. `--keep-work` retains additional diagnostics. Return the entire successful output directory for the article comparison. On failure, return `run.json`, `verification.json` and retained diagnostics; a failed run supplies no replacement article numbers.

The two `run.json` files have different roles: the output-root file records the experiment; `inputs/run.json` holds numerical settings and is bundled inside the proof. The five defining inputs are `reference.txt`, `transport.txt`, `profile.txt`, `radius.txt` and that configuration. `provenance.json` binds them by SHA-256, without proving their numerical validity.

`verification.json → numerical_results` has one complete numerical tree:

| Subtree | Contents |
|---|---|
| `intervals.reference` | Normalization, base, masses and comparison values/signals |
| `intervals.certificate` | Signed dual bound and total price |
| `intervals.feasible` | Radius, norms, central energy, feasible ceiling, six `F_i`, 36 `J_i_j` |
| `parameters_exact` | Reference data, potential terms/multipliers, primal coefficients/directions/preconditioner; sine truncation `J` |
| `contraction_exact` | Recomputed rational residual norm, contraction and energy ceiling |
| `bound_terms` | Four signed observation contributions before common dilation |
| `sine_recomputed` | Rational enclosures of sum, tail, energy and constants |
| `derived` | Gain/baseline ratios, reference-to-sine interval, feasible improvement lower bound and primal–dual gap upper bound |
| `counts`, `cover_geometry` | Cells/leaves, worst recorded bound, dilation and exact price bookkeeping |

Intervals have `exact` rational endpoints and `decimal_outward` endpoints. Code/paper residual indices differ by one. Derived scalar improvement/gap endpoints are an outward presentation of an already conservative scalar bound. There is no duplicate `results.json`, separate bound assembled by another program, or publication-specific table with hard-coded digits.

`run.json.phases` records search, construction, packaging and verification wall times (or input copying instead of search). Resume retains prior phase records; `wall_seconds_this_invocation` covers only the latest invocation. `verification.json.elapsed_seconds` covers the internal standalone check, including compilation, but excludes extraction/cleanup. `archive_check_seconds` includes outer extraction/launch/cleanup; `archive_extraction_seconds` identifies extraction. Native durations overlap and must not be summed as wall time. `toolchain.compilation_seconds` records compilation separately. Machine memory capacity is recorded when available; peak process memory is not measured.
