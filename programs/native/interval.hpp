#pragma once
#include "mpfr_api.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <limits>
#include <sstream>

// Atomic directed rounding: L. Fousse et al., "MPFR: A multiple-precision
// binary floating-point library with correct rounding", ACM TOMS 33(2),
// Article 13 (2007), DOI 10.1145/1236463.1236468. Author PDF sections 2.1
// (p. 3) and 2.3 (p. 4):
// https://perso.ens-lyon.fr/guillaume.hanrot/Papers/toms.pdf
// Composition and all differential error bounds are proved in the analytical justification,
// sec:enclosures, lem:jets, and lem:flow-step; MPFR supplies
// only the explicitly directed atomic operations used here.

namespace cert {
inline long precision = 64;
inline void require(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}

// A failed write is an I/O failure, never a refinable interval enclosure.
inline std::ofstream output_file(const std::string &path) {
    std::ofstream out;
    out.exceptions(std::ios::failbit|std::ios::badbit);
    out.open(path);
    return out;
}
inline void finish_output(std::ofstream &out) { out.flush(); out.close(); }
inline unsigned read_count(std::istream &in) {
    std::string token; in>>token;
    require(bool(in)&&!token.empty(),"missing integer dimension");
    require(token=="0"||token.front()!='0',"noncanonical integer dimension");
    unsigned value=0;
    for(char c:token) {
        require(c>='0'&&c<='9',"nonnegative integer dimension required");
        unsigned digit=unsigned(c-'0');
        require(value<=(std::numeric_limits<unsigned>::max()-digit)/10,"dimension overflow");
        value=10*value+digit;
    }
    return value;
}
inline void require_end(std::istream &in) {
    require(!in.fail()&&!in.bad(),"truncated numerical data");
    in>>std::ws;
    require(in.eof()&&!in.bad(),"unexpected trailing numerical data");
}
inline unsigned count_argument(const char *text) {
    std::istringstream in(text);
    unsigned value=read_count(in);require_end(in);return value;
}

struct Real {
    mpfr_t x;
    Real() { mpfr_init2(x, precision); mpfr_set_si(x, 0, MPFR_RNDN); }
    Real(const Real &a) {
        mpfr_init2(x, mpfr_get_prec(a.x)); mpfr_set(x, a.x, MPFR_RNDN);
    }
    Real(Real &&a) noexcept : Real() { mpfr_swap(x, a.x); }
    ~Real() { mpfr_clear(x); }
    Real &operator=(const Real &a) {
        if (this != &a) {
            if(mpfr_get_prec(x)!=mpfr_get_prec(a.x))mpfr_set_prec(x, mpfr_get_prec(a.x));
            mpfr_set(x, a.x, MPFR_RNDN);
        }
        return *this;
    }
    Real &operator=(Real &&a) noexcept { mpfr_swap(x, a.x); return *this; }
};

/* Invariant: finite exact reals lo<=hi. Every arithmetic operation rounds
 * its lower endpoint down and its upper endpoint up. No proof condition
 * uses a nearest-rounded decimal or a midpoint as an enclosure endpoint.
 */
struct I {
    Real lo, hi;
    I() = default;
    I(int n) {
        mpfr_set_si(lo.x, n, MPFR_RNDD); mpfr_set_si(hi.x, n, MPFR_RNDU);
    }
    explicit I(double n) {
        require(std::isfinite(n), "nonfinite binary input");
        mpfr_set_d(lo.x, n, MPFR_RNDD); mpfr_set_d(hi.x, n, MPFR_RNDU);
    }
    explicit I(const std::string &s) {
        require(!mpfr_set_str(lo.x, s.c_str(), 0, MPFR_RNDD), "invalid real");
        require(!mpfr_set_str(hi.x, s.c_str(), 0, MPFR_RNDU), "invalid real");
        check();
    }
    void check() const {
        require(mpfr_number_p(lo.x) && mpfr_number_p(hi.x), "nonfinite interval");
        require(mpfr_cmp(lo.x, hi.x) <= 0, "reversed interval");
    }
    bool positive() const { return mpfr_number_p(lo.x) && mpfr_cmp_si(lo.x, 0) > 0; }
    bool nonnegative() const { return mpfr_number_p(lo.x) && mpfr_cmp_si(lo.x, 0) >= 0; }
    bool zero() const {
        return mpfr_number_p(lo.x)&&mpfr_number_p(hi.x)
               &&mpfr_cmp_si(lo.x,0)==0&&mpfr_cmp_si(hi.x,0)==0;
    }
    double lower_double() const { return mpfr_get_d(lo.x, MPFR_RNDD); }
    double upper_double() const { return mpfr_get_d(hi.x, MPFR_RNDU); }
};

inline I lower(const I &a) {
    I r; r.lo=a.lo; r.hi=a.lo; return r;
}
inline I upper(const I &a) {
    I r; r.lo=a.hi; r.hi=a.hi; return r;
}
inline I hull(const I &a, const I &b) {
    I r;
    mpfr_set(r.lo.x, mpfr_cmp(a.lo.x,b.lo.x)<0 ? a.lo.x:b.lo.x, MPFR_RNDD);
    mpfr_set(r.hi.x, mpfr_cmp(a.hi.x,b.hi.x)>0 ? a.hi.x:b.hi.x, MPFR_RNDU);
    return r;
}
inline bool contained(const I &a, const I &b) {
    return mpfr_cmp(a.lo.x,b.lo.x)>=0 && mpfr_cmp(a.hi.x,b.hi.x)<=0;
}
inline I intersect(const I &a, const I &b) {
    I r;
    mpfr_set(r.lo.x, mpfr_cmp(a.lo.x,b.lo.x)>0 ? a.lo.x:b.lo.x, MPFR_RNDD);
    mpfr_set(r.hi.x, mpfr_cmp(a.hi.x,b.hi.x)<0 ? a.hi.x:b.hi.x, MPFR_RNDU);
    r.check(); return r;
}
inline I operator+(const I &a, const I &b) {
    I r; mpfr_add(r.lo.x,a.lo.x,b.lo.x,MPFR_RNDD);
    mpfr_add(r.hi.x,a.hi.x,b.hi.x,MPFR_RNDU); return r;
}
inline I operator-(const I &a, const I &b) {
    I r; mpfr_sub(r.lo.x,a.lo.x,b.hi.x,MPFR_RNDD);
    mpfr_sub(r.hi.x,a.hi.x,b.lo.x,MPFR_RNDU); return r;
}
inline I operator-(const I &a) {
    I r; mpfr_neg(r.lo.x,a.hi.x,MPFR_RNDD);
    mpfr_neg(r.hi.x,a.lo.x,MPFR_RNDU); return r;
}
inline I operator*(const I &a, const I &b) {
    // Bilinearity puts extrema at corners. Sign classes identify two
    // corners, except when both intervals straddle zero (four corners).
    I r;
    const bool ap=a.nonnegative(), an=mpfr_cmp_si(a.hi.x,0)<=0;
    const bool bp=b.nonnegative(), bn=mpfr_cmp_si(b.hi.x,0)<=0;
    mpfr_srcptr l1,l2,h1,h2;
    if(ap) {
        l1=bp?a.lo.x:a.hi.x;l2=b.lo.x;
        h1=bn?a.lo.x:a.hi.x;h2=b.hi.x;
    } else if(an) {
        l1=bn?a.hi.x:a.lo.x;l2=b.hi.x;
        h1=bp?a.hi.x:a.lo.x;h2=b.lo.x;
    } else if(bp) {
        l1=a.lo.x;l2=b.hi.x;h1=a.hi.x;h2=b.hi.x;
    } else if(bn) {
        l1=a.hi.x;l2=b.lo.x;h1=a.lo.x;h2=b.lo.x;
    } else {
        Real temp;
        mpfr_mul(r.lo.x,a.lo.x,b.hi.x,MPFR_RNDD);
        mpfr_mul(temp.x,a.hi.x,b.lo.x,MPFR_RNDD);
        if(mpfr_cmp(temp.x,r.lo.x)<0)r.lo=temp;
        mpfr_mul(r.hi.x,a.lo.x,b.lo.x,MPFR_RNDU);
        mpfr_mul(temp.x,a.hi.x,b.hi.x,MPFR_RNDU);
        if(mpfr_cmp(temp.x,r.hi.x)>0)r.hi=temp;
        return r;
    }
    mpfr_mul(r.lo.x,l1,l2,MPFR_RNDD);
    mpfr_mul(r.hi.x,h1,h2,MPFR_RNDU);
    return r;
}
inline I inverse(const I &a) {
    require(a.positive() || mpfr_cmp_si(a.hi.x,0)<0, "division interval contains zero");
    I r, one(1);
    mpfr_div(r.lo.x,one.lo.x,a.hi.x,MPFR_RNDD);
    mpfr_div(r.hi.x,one.hi.x,a.lo.x,MPFR_RNDU); return r;
}
inline I operator/(const I &a, const I &b) { return a*inverse(b); }
inline I square(const I &a) {
    I r; Real x, y;
    mpfr_mul(x.x,a.lo.x,a.lo.x,MPFR_RNDU);
    mpfr_mul(y.x,a.hi.x,a.hi.x,MPFR_RNDU);
    mpfr_set(r.hi.x,mpfr_cmp(x.x,y.x)>0?x.x:y.x,MPFR_RNDU);
    if (a.nonnegative()) mpfr_mul(r.lo.x,a.lo.x,a.lo.x,MPFR_RNDD);
    else if (mpfr_cmp_si(a.hi.x,0)<=0) mpfr_mul(r.lo.x,a.hi.x,a.hi.x,MPFR_RNDD);
    return r;
}
inline I abs_bound(const I &a) {
    I r; Real x, y;
    mpfr_abs(x.x,a.lo.x,MPFR_RNDU); mpfr_abs(y.x,a.hi.x,MPFR_RNDU);
    mpfr_set(r.lo.x,mpfr_cmp(x.x,y.x)>0?x.x:y.x,MPFR_RNDU);
    mpfr_set(r.hi.x,r.lo.x,MPFR_RNDN); return r;
}
inline I pow(I a, unsigned n) {
    I r(1);
    while(n) { if(n&1) r=r*a; n>>=1; if(n) a=square(a); }
    return r;
}
inline I sqrt(const I &a) {
    require(a.nonnegative(), "negative square-root argument");
    I r; mpfr_sqrt(r.lo.x,a.lo.x,MPFR_RNDD); mpfr_sqrt(r.hi.x,a.hi.x,MPFR_RNDU); return r;
}
inline I cbrt(const I &a) {
    I r; mpfr_cbrt(r.lo.x,a.lo.x,MPFR_RNDD); mpfr_cbrt(r.hi.x,a.hi.x,MPFR_RNDU); return r;
}
inline I exp(const I &a) {
    I r; mpfr_exp(r.lo.x,a.lo.x,MPFR_RNDD); mpfr_exp(r.hi.x,a.hi.x,MPFR_RNDU); return r;
}
inline I log(const I &a) {
    require(a.positive(), "nonpositive logarithm argument");
    I r; mpfr_log(r.lo.x,a.lo.x,MPFR_RNDD); mpfr_log(r.hi.x,a.hi.x,MPFR_RNDU); return r;
}
inline I pi() {
    I r; mpfr_const_pi(r.lo.x,MPFR_RNDD); mpfr_const_pi(r.hi.x,MPFR_RNDU); return r;
}
inline I midpoint(const I &a) {
    a.check();
    I m=lower((lower(a)+upper(a))/I(2));
    // Keep an exact point inside a even if its endpoints were copied from
    // a higher precision. The subsequent radius includes the asymmetry.
    if(mpfr_cmp(m.lo.x,a.lo.x)<0)return lower(a);
    if(mpfr_cmp(m.hi.x,a.hi.x)>0)return upper(a);
    return m;
}
inline I radius(const I &a, const I &m) { return abs_bound(a-m); }
inline I centered(const I &a) { return hull(-abs_bound(a),abs_bound(a)); }
inline I width(const I &a) { return upper(a)-lower(a); }

inline std::string hex_real(mpfr_srcptr a) {
    int n=mpfr_snprintf(nullptr,0,"%Ra",a);
    require(n>0,"MPFR output failure");
    std::vector<char> s(n+1); mpfr_snprintf(s.data(),s.size(),"%Ra",a);
    return std::string(s.data());
}
inline std::ostream &operator<<(std::ostream &os,const I &a) {
    a.check(); return os<<hex_real(a.lo.x)<<" "<<hex_real(a.hi.x);
}
inline std::istream &operator>>(std::istream &is,I &a) {
    std::string l,u;
    if(is>>l>>u) {
        I lower_input(l),upper_input(u);
        mpfr_set(a.lo.x,lower_input.lo.x,MPFR_RNDD);
        mpfr_set(a.hi.x,upper_input.hi.x,MPFR_RNDU);
        a.check();
    }
    return is;
}
inline void check_runtime() {
    require(std::string(mpfr_get_version()).at(0)=='4',"this build requires MPFR 4");
    I a("0x1.0000000000000001p0");
    require(mpfr_get_prec(a.lo.x)==precision,"MPFR precision/ABI mismatch");
    require(contained(a,I(1)+I(1)/pow(I(2),64)),"MPFR ABI arithmetic mismatch");
    require(!mpfr_underflow_p()&&!mpfr_overflow_p()&&!mpfr_nanflag_p()&&!mpfr_divby0_p()&&!mpfr_erangeflag_p(),
            "MPFR exception flag set");
}

/* Monotonicity reduces interval cubic roots to endpoint roots. Interval
 * Newton is applied only where f'(x)>0; every intersection contains the
 * unique nonnegative root. Rounding stagnation is accepted because the
 * returned enclosure, unlike its midpoint, is already mathematically valid.
 */
inline I cubic_point(const I &a,const I &b) {
    require(a.nonnegative() && b.positive(),"invalid positive cubic parameters");
    I x=hull(lower(sqrt(a)),upper(sqrt(a)+cbrt(b)));
    x=intersect(x,hull(lower(cbrt(b)),upper(x)));
    for(long n=0;n<precision;n++) {
        I den=I(3)*square(x)-a;
        if(!den.positive()) break;
        I m=midpoint(x), old=x;
        x=intersect(x,m-((square(m)-a)*m-b)/den);
        if(mpfr_cmp(width(x).hi.x,width(old).hi.x)>=0) break;
    }
    return x;
}
inline I cubic(const I &a,const I &b) {
    I l=cubic_point(lower(a),lower(b));
    I h=cubic_point(upper(a),upper(b));
    return hull(lower(l),upper(h));
}
} // namespace cert
