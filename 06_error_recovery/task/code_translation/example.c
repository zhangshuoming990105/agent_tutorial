#include <stdio.h>
double interp_weno7(double phim3, double phim2, double phim1, double phi,
    double phip1, double phip2, double phip3) {
const double p0 = (-1.0 / 4.0) * phim3 + (13.0 / 12.0) * phim2 +
    (-23.0 / 12.0) * phim1 + (25.0 / 12.0) * phi;
const double p1 = (1.0 / 12.0) * phim2 + (-5.0 / 12.0) * phim1 +
    (13.0 / 12.0) * phi + (1.0 / 4.0) * phip1;
const double p2 = (-1.0 / 12.0) * phim1 + (7.0 / 12.0) * phi +
    (7.0 / 12.0) * phip1 + (-1.0 / 12.0) * phip2;
const double p3 = (1.0 / 4.0) * phi + (13.0 / 12.0) * phip1 +
    (-5.0 / 12.0) * phip2 + (1.0 / 12.0) * phip3;
const double beta0 =
(phim3 *
(547.0 * phim3 - 3882.0 * phim2 + 4642.0 * phim1 - 1854.0 * phi) +
phim2 * (7043.0 * phim2 - 17246.0 * phim1 + 7042.0 * phi) +
phim1 * (11003.0 * phim1 - 9402.0 * phi) + 2107.0 * phi * phi);
const double beta1 =
(phim2 * (267.0 * phim2 - 1642.0 * phim1 + 1602.0 * phi - 494.0 * phip1) +
phim1 * (2843.0 * phim1 - 5966.0 * phi + 1922.0 * phip1) +
phi * (3443.0 * phi - 2522.0 * phip1) + 547.0 * phip1 * phip1);
const double beta2 =
(phim1 * (547.0 * phim1 - 2522.0 * phi + 1922.0 * phip1 - 494.0 * phip2) +
phi * (3443.0 * phi - 5966.0 * phip1 + 1602.0 * phip2) +
phip1 * (2843.0 * phip1 - 1642.0 * phip2) + 267.0 * phip2 * phip2);
const double beta3 =
(phi * (2107.0 * phi - 9402.0 * phip1 + 7042.0 * phip2 - 1854.0 * phip3) +
phip1 * (11003.0 * phip1 - 17246.0 * phip2 + 4642.0 * phip3) +
phip2 * (7043.0 * phip2 - 3882.0 * phip3) + 547.0 * phip3 * phip3);
const double alpha0 = (1.0 / 35.0) / ((beta0 + 1e-10) * (beta0 + 1e-10));
const double alpha1 = (12.0 / 35.0) / ((beta1 + 1e-10) * (beta1 + 1e-10));
const double alpha2 = (18.0 / 35.0) / ((beta2 + 1e-10) * (beta2 + 1e-10));
const double alpha3 = (4.0 / 35.0) / ((beta3 + 1e-10) * (beta3 + 1e-10));
const double alpha_sum_inv = 1.0 / (alpha0 + alpha1 + alpha2 + alpha3);
const double w0 = alpha0 * alpha_sum_inv;
const double w1 = alpha1 * alpha_sum_inv;
const double w2 = alpha2 * alpha_sum_inv;
const double w3 = alpha3 * alpha_sum_inv;
return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
}

int main() {
    double phim3 = 3.2;
    double phim2 = 2.1;
    double phim1 = 1.0;
    double phi = 0.5;
    double phip1 = 1.5;
    double phip2 = 2.5;
    double phip3 = 3.5;
    double result = interp_weno7(phim3, phim2, phim1, phi, phip1, phip2, phip3);
    printf("Result: %f\n", result);
    return 0;   
}