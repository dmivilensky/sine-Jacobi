#pragma once
#include <mpfr.h>

// MPFR 4.2.2 manual, "Headers and Libraries" and "Rounding":
// https://www.mpfr.org/mpfr-4.2.2/mpfr.html
// Use the public declarations; no handwritten ABI or private layout.
static_assert(MPFR_VERSION_MAJOR == 4, "MPFR 4 development headers required");
