#pragma once
// Normalized derivatives f_j=f^(j)/j! and Cauchy coefficient algebra.
// See lem:jets for triangular identities, and lem:quadrature /
// lem:flow-step for the corresponding remainder bounds.
// Taylor coefficient generation in validated ODE methods is surveyed by
// Nedialkov, Jackson and Corliss (1999), DOI 10.1016/S0096-3003(98)10083-8,
// Section 2. No ordinary solver tolerances enter these enclosures.
#include "interval.hpp"

namespace cert {
/* A jet stores f^(j)/j!. The same recurrences are valid for intervals
 * enclosing these derivatives at every point of a whole real segment.
 * Products use Cauchy convolution; inverse and implicit-root coefficients
 * solve triangular coefficient equations. These are proved algebraically
 * in the analytical derivation, so no automatic-differentiation theorem is assumed.
 */
struct Jet {
    std::vector<I> c;
    explicit Jet(unsigned n=0,I value=I(0)) : c(n+1) { c[0]=value; }
    unsigned order() const { return c.size()-1; }
    I &operator[](unsigned n) { return c.at(n); }
    const I &operator[](unsigned n) const { return c.at(n); }
};
inline Jet variable(const I &t,unsigned n) {
    Jet r(n,t); if(n) r[1]=I(1); return r;
}
inline Jet operator+(const Jet &a,const Jet &b) {
    require(a.order()==b.order(),"jet order mismatch"); Jet r(a.order());
    for(unsigned i=0;i<=r.order();i++) r[i]=a[i]+b[i];
    return r;
}
inline Jet operator-(const Jet &a,const Jet &b) {
    require(a.order()==b.order(),"jet order mismatch"); Jet r(a.order());
    for(unsigned i=0;i<=r.order();i++) r[i]=a[i]-b[i];
    return r;
}
inline Jet operator-(const Jet &a) { return Jet(a.order())-a; }
inline Jet operator*(const Jet &a,const I &b) {
    Jet r(a.order()); for(unsigned i=0;i<=r.order();i++) r[i]=a[i]*b; return r;
}
inline Jet operator*(const I &a,const Jet &b) { return b*a; }
inline Jet operator/(const Jet &a,const I &b) { return a*inverse(b); }
inline Jet operator*(const Jet &a,const Jet &b) {
    require(a.order()==b.order(),"jet order mismatch"); Jet r(a.order());
    for(unsigned i=0;i<=r.order();i++)
        for(unsigned j=0;j<=i;j++) r[i]=r[i]+a[j]*b[i-j];
    return r;
}
inline Jet inverse(const Jet &a) {
    Jet r(a.order(),inverse(a[0]));
    for(unsigned i=1;i<=r.order();i++) {
        I v;
        for(unsigned j=1;j<=i;j++) v=v+a[j]*r[i-j];
        r[i]=-v*r[0];
    }
    return r;
}
inline Jet operator/(const Jet &a,const Jet &b) { return a*inverse(b); }
inline Jet square(const Jet &a) {
    Jet r(a.order());
    // Pair equal Cauchy terms. The diagonal is the square of the same
    // derivative and therefore has a nonnegative enclosure.
    for(unsigned n=0;n<=a.order();n++) {
        for(unsigned j=0;j<(n+1)/2;j++)r[n]=r[n]+I(2)*a[j]*a[n-j];
        if(n%2==0)r[n]=r[n]+square(a[n/2]);
    }
    return r;
}
inline Jet nonnegative_value(Jet a) {
    require(mpfr_cmp_si(a[0].hi.x,0)>=0,"enclosure contradicts analytic nonnegativity");
    a[0]=intersect(a[0],hull(I(0),upper(a[0])));
    return a;
}
inline Jet cube(const Jet &a) { return square(a)*a; }
inline Jet pow(Jet a,unsigned n) {
    Jet r(a.order(),I(1));
    while(n) {if(n&1)r=r*a;n>>=1;if(n)a=square(a);}
    return r;
}
inline Jet exp(const Jet &a) {
    Jet r(a.order(),exp(a[0]));
    // r'=a'r, hence n r_n=sum_{j=1}^n j a_j r_{n-j}.
    for(unsigned n=1;n<=r.order();n++) {
        I value;
        for(unsigned j=1;j<=n;j++) value=value+I(int(j))*a[j]*r[n-j];
        r[n]=value/I(int(n));
    }
    return r;
}
inline Jet sqrt(const Jet &a) {
    require(a[0].positive(),"positive square-root jet required");
    Jet r(a.order(),sqrt(a[0]));
    // Compare coefficients in r^2=a; the new coefficient occurs twice.
    for(unsigned n=1;n<=r.order();n++) {
        I value=a[n];
        for(unsigned j=1;j<n;j++) value=value-r[j]*r[n-j];
        r[n]=value/(I(2)*r[0]);
    }
    return r;
}
inline Jet cubic(const Jet &a,const Jet &b) {
    require(a.order()==b.order(),"jet order mismatch");
    Jet r(a.order(),cubic(a[0],b[0]));
    // At the root, 3*z^2-a=2*a+3*b/z. This positive expression avoids
    // a false singularity caused by subtracting correlated intervals.
    I den=I(2)*a[0]+I(3)*b[0]/r[0];
    require(den.positive(),"implicit-root derivative not positive");
    const I reciprocal=inverse(den);
    std::vector<I> square_without_constant(r.order()+1);
    for(unsigned i=1;i<=r.order();i++) {
        // Write z=z0+u, u0=0. The known part of [z^3]_i equals
        // 3*z0*[u^2]_i+[u^3]_i. Reuse [u^2]_i to obtain O(N^2)
        // interval operations through order N, with the same identity.
        I v=b[i];
        for(unsigned j=1;j<=i;j++) v=v+a[j]*r[i-j];
        I &sq=square_without_constant[i];
        for(unsigned j=1;j<(i+1)/2;j++)sq=sq+I(2)*r[j]*r[i-j];
        if(i%2==0)sq=sq+square(r[i/2]);
        I known=I(3)*r[0]*sq;
        for(unsigned j=1;j+1<i;j++)known=known+r[j]*square_without_constant[i-j];
        r[i]=(v-known)*reciprocal;
    }
    return r;
}
inline I evaluate(const Jet &p,const I &x,unsigned degree) {
    I r=p[degree];
    while(degree) r=p[--degree]+r*x;
    return r;
}
} // namespace cert
