#pragma once
// Binary64 interval values, gradients and Hessians on complete boxes.
// The scalar rounding contract is sec:enclosures. The conjugate is
// lem:conjugate; whole-box Taylor bounds are lem:box-bound.
// For interval verification methods see Rump, Acta Numerica 19 (2010),
// 287--449, DOI 10.1017/S096249291000005X.
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace boxes {
inline void require(bool ok,const char *message) {
    if(!ok) throw std::runtime_error(message);
}
// IEEE binary64 magnitudes are ordered by their unsigned encodings.
// Advancing one encoding is exactly nextafter, including subnormals.
// memcpy avoids aliasing assumptions; overflow is rejected by D::check.
inline double up(double x) {
    if(std::isnan(x)||x==INFINITY)return x;
    if(x==0)return std::numeric_limits<double>::denorm_min();
    uint64_t bits;std::memcpy(&bits,&x,sizeof(bits));
    if(x>0)++bits;else --bits;
    double result;std::memcpy(&result,&bits,sizeof(result));return result;
}
inline double down(double x) { return -up(-x); }
struct D {
    double lo=0,hi=0;
    D()=default;
    D(int x):lo(x),hi(x){}
    explicit D(double x):lo(x),hi(x) { require(std::isfinite(x),"nonfinite point"); }
    D(double a,double b):lo(a),hi(b) { check(); }
    void check() const { require(std::isfinite(lo)&&std::isfinite(hi)&&lo<=hi,"invalid box interval"); }
};
inline bool zero(D a) { return a.lo==0&&a.hi==0; }
inline D operator+(D a,D b) {
    if(zero(a))return b;
    if(zero(b))return a;
    return D(down(a.lo+b.lo),up(a.hi+b.hi));
}
inline D operator-(D a,D b) { return D(down(a.lo-b.hi),up(a.hi-b.lo)); }
inline D operator-(D a) { return D(-a.hi,-a.lo); }
inline D operator*(D a,D b) {
    if(zero(a)||zero(b))return D(0);
    double x[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
    return D(down(*std::min_element(x,x+4)),up(*std::max_element(x,x+4)));
}
inline D inverse(D a) {
    require(a.lo>0||a.hi<0,"division interval contains zero");
    return D(down(1/a.hi),up(1/a.lo));
}
inline D operator/(D a,D b) { return a*inverse(b); }
inline D square(D a) {
    if(zero(a))return D(0);
    if(a.lo>=0) return D(std::max(0.,down(a.lo*a.lo)),up(a.hi*a.hi));
    if(a.hi<=0) return D(std::max(0.,down(a.hi*a.hi)),up(a.lo*a.lo));
    return D(0,up(std::max(a.lo*a.lo,a.hi*a.hi)));
}
inline D hull(D a,D b) { return D(std::min(a.lo,b.lo),std::max(a.hi,b.hi)); }
inline D positive_part(D a) { return D(std::max(0.,a.lo),std::max(0.,a.hi)); }
inline double magnitude(D a) { return std::max(std::abs(a.lo),std::abs(a.hi)); }
inline D pow(D a,unsigned n) {
    D r(1);
    while(n) { if(n&1) r=r*a; n>>=1; if(n) a=square(a); }
    return r;
}
inline D positive_floor(D a,D lower) {
    a.lo=std::max(a.lo,lower.lo); a.check(); return a;
}

inline void check_runtime() {
    require(std::numeric_limits<double>::is_iec559&&sizeof(double)==8&&sizeof(uint64_t)==8&&FLT_RADIX==2&&DBL_MANT_DIG==53
            &&DBL_MAX_EXP==1024&&FLT_EVAL_METHOD==0,"binary64 environment required");
    require(std::fegetround()==FE_TONEAREST,"round-to-nearest environment required");
    double one=1.; uint64_t bits; std::memcpy(&bits,&one,sizeof(bits));
    require(bits==UINT64_C(0x3ff0000000000000),"unexpected binary64 encoding");
    volatile double least_normal=std::numeric_limits<double>::min();
    volatile double least_subnormal=std::numeric_limits<double>::denorm_min();
    volatile double two=2.,one_factor=1.;
    require(least_normal/two>0&&least_subnormal*one_factor==least_subnormal
            &&least_subnormal+least_subnormal==up(least_subnormal),
            "gradual underflow without flushed subnormal operands is required");
}

/* A carries the value, gradient, and full symmetric Hessian with respect
 * to (U,T,t). Every entry encloses its exact derivative on the whole box.
 * Only +,-,*,inverse and the C^2 positive-cube composition are needed.
 * Their derivative rules are elementary algebra, used in the analytical derivation.
 */
struct A {
    D v;
    std::array<D,3> g{};
    std::array<std::array<D,3>,3> h{};
    A()=default;
    A(D value):v(value){}
    A(int value):v(value){}
};
inline A variable(D x,unsigned i) { A a(x); a.g.at(i)=D(1); return a; }
inline A operator+(const A &a,const A &b) {
    A r(a.v+b.v);
    for(unsigned i=0;i<3;i++) {
        r.g[i]=a.g[i]+b.g[i];
        for(unsigned j=i;j<3;j++) r.h[j][i]=r.h[i][j]=a.h[i][j]+b.h[i][j];
    }
    return r;
}
inline A operator-(const A &a) {
    A r(-a.v);
    for(unsigned i=0;i<3;i++) {
        r.g[i]=-a.g[i];
        for(unsigned j=0;j<3;j++) r.h[i][j]=-a.h[i][j];
    }
    return r;
}
inline A operator-(const A &a,const A &b) { return a+(-b); }
inline A operator*(const A &a,const A &b) {
    A r(a.v*b.v);
    for(unsigned i=0;i<3;i++) {
        r.g[i]=a.g[i]*b.v+a.v*b.g[i];
        for(unsigned j=i;j<3;j++)
            r.h[j][i]=r.h[i][j]=a.h[i][j]*b.v+a.g[i]*b.g[j]+a.g[j]*b.g[i]+a.v*b.h[i][j];
    }
    return r;
}
inline A operator*(const A &a,D b) {
    A r(a.v*b);
    for(unsigned i=0;i<3;i++) {
        r.g[i]=a.g[i]*b;
        for(unsigned j=i;j<3;j++)r.h[j][i]=r.h[i][j]=a.h[i][j]*b;
    }
    return r;
}
inline A operator*(D a,const A &b) { return b*a; }
inline A inverse(const A &a) {
    A r(inverse(a.v)); D inv2=square(r.v),inv3=inv2*r.v;
    for(unsigned i=0;i<3;i++) {
        r.g[i]=-a.g[i]*inv2;
        for(unsigned j=i;j<3;j++)
            r.h[j][i]=r.h[i][j]=D(2)*a.g[i]*a.g[j]*inv3-a.h[i][j]*inv2;
    }
    return r;
}
inline A operator/(const A &a,const A &b) { return a*inverse(b); }
inline A operator/(const A &a,D b) { return a*inverse(b); }
inline A square(const A &a) { A r=a*a; r.v=square(a.v); return r; }
inline A pow(A a,unsigned n) {
    A r(1);
    while(n) { if(n&1) r=r*a; n>>=1; if(n) a=square(a); }
    return r;
}
inline A cubic_conjugate(const A &z) {
    // psi(z)=-z-1/2+(4/27)*(z+3/2)_+^3. First and second derivatives
    // match at -3/2, so no unexamined branch or nonsmooth Hessian is hidden.
    D u=positive_part(z.v+D(3)/D(2));
    D first=D(-1)+(D(4)/D(9))*square(u);
    D second=(D(8)/D(9))*u;
    D value;
    auto upper_branch=[](D x) { return square(x)*(D(2)/D(3)+(D(4)/D(27))*x); };
    if(z.v.lo>=-1.5) value=upper_branch(z.v);
    else if(z.v.hi<=-1.5) value=-z.v-D(1)/D(2);
    else value=hull(-D(z.v.lo,-1.5)-D(1)/D(2),upper_branch(D(-1.5,z.v.hi)));
    A r(value);
    for(unsigned i=0;i<3;i++) {
        r.g[i]=first*z.g[i];
        for(unsigned j=i;j<3;j++)
            r.h[j][i]=r.h[i][j]=second*z.g[i]*z.g[j]+first*z.h[i][j];
    }
    return r;
}
inline A polynomial_horner(const std::vector<D> &coefficients,const A &q) {
    require(!coefficients.empty(),"empty polynomial");
    A r(coefficients.back());
    for(std::size_t j=coefficients.size()-1;j>0;j--) r=r*q+A(coefficients[j-1]);
    return r;
}
inline A intersect(const A &a,const A &b) {
    auto meet=[](D x,D y) {return D(std::max(x.lo,y.lo),std::min(x.hi,y.hi));};
    A out(meet(a.v,b.v));
    for(unsigned i=0;i<3;i++) {
        out.g[i]=meet(a.g[i],b.g[i]);
        for(unsigned j=0;j<3;j++)out.h[i][j]=meet(a.h[i][j],b.h[i][j]);
    }
    return out;
}
inline A polynomial(const std::vector<D> &coefficients,const A &q) {
    // Exact finite Taylor translation, not a truncated approximation:
    // p(q)=sum_j p^(j)(c)/j! * (q-c)^j. Repeated synthetic translation
    // computes all coefficients with directed intervals. Evaluating both
    // forms and intersecting each jet component preserves enclosure and
    // reduces cancellation from an alternating power-basis polynomial.
    // Proof: the analytical justification, Lemma lem:translation.
    A direct=polynomial_horner(coefficients,q);
    double center=q.v.lo+(q.v.hi-q.v.lo)/2;
    D c(center);std::vector<D> shifted=coefficients;
    for(std::size_t first=shifted.size()-1;first>0;--first)
        for(std::size_t j=first-1;j+1<shifted.size();++j)
            shifted[j]=shifted[j]+c*shifted[j+1];
    A translated=polynomial_horner(shifted,q-A(c));
    auto intersect=[](D a,D b) {return D(std::max(a.lo,b.lo),std::min(a.hi,b.hi));};
    direct.v=intersect(direct.v,translated.v);
    for(unsigned i=0;i<3;i++) {
        direct.g[i]=intersect(direct.g[i],translated.g[i]);
        for(unsigned j=0;j<3;j++)direct.h[i][j]=intersect(direct.h[i][j],translated.h[i][j]);
    }
    return direct;
}
} // namespace boxes
