#pragma once
// Evaluate the normalized storage inequality and all its derivatives.
// Exact cancellation follows lem:regular-criterion. State dependence is
// separated from spatial coefficients by an algebraic factorization;
// lem:box-bound and lem:mobius justify tightening
// each component by intersection of two enclosures of the same function.
#include "reference.hpp"
#include "box_arithmetic.hpp"
#include "rational.hpp"

namespace transport {
using boxes::A;
using boxes::D;
using cert::I;
using cert::Jet;

inline D convert(const I &x) { return D(x.lower_double(),x.upper_double()); }
inline I read_rational(std::istream &in,cert::Rational *exact=nullptr) {
    std::string token; in>>token; cert::require(bool(in),"missing rational input");
    cert::Rational value(token);
    if(exact)*exact=value;
    auto slash=token.find('/');
    if(slash==std::string::npos) return I(token);
    cert::require(token.find('/',slash+1)==std::string::npos,"invalid rational input");
    I den(token.substr(slash+1)); cert::require(den.positive(),"nonpositive rational denominator");
    return I(token.substr(0,slash))/den;
}
inline A profile_jet(const Jet &j) {
    cert::require(j.order()>=2,"second-order profile data required");
    A a(convert(j[0])); a.g[2]=convert(j[1]); a.h[2][2]=convert(I(2)*j[2]);
    return a;
}
inline A unit_value(A a) {
    a.v.lo=std::max(a.v.lo,0.); a.v.hi=std::min(a.v.hi,1.); a.v.check();
    return a;
}

inline void centered_spatial_jet(A &whole,const A &middle,D delta) {
    // Every coefficient here is a function of t alone. Taylor's integral
    // formula at the fixed midpoint encloses its value and first derivative
    // without carrying the cancellation width of its defining expression.
    // The original and Taylor intervals are both retained by intersection.
    // Taylor's formula in t, also used in lem:box-bound(ii); the
    // first-derivative enclosure follows from the mean value theorem.
    auto intersect=[](D a,D b) {return D(std::max(a.lo,b.lo),std::min(a.hi,b.hi));};
    whole.v=intersect(whole.v,middle.v+middle.g[2]*delta
                                  +whole.h[2][2]*boxes::square(delta)/D(2));
    whole.g[2]=intersect(whole.g[2],middle.g[2]+whole.h[2][2]*delta);
}

struct StorageTerm {
    bool moment;
    unsigned power0,power1;
    D coefficient;
    std::array<std::array<std::vector<D>,4>,2> quotients;
};
struct Parameters {
    D dilation=D(1);
    std::array<D,2> scales;
    std::array<D,4> multipliers;
    std::vector<StorageTerm> terms;
    explicit Parameters(const std::string &path) {
        // Check the endpoint and derivative identities over Q here,
        // independently of Python's polynomial construction.
        std::ifstream in(path); cert::require(bool(in),"cannot read transport polynomial");
        std::string magic;in>>magic;
        unsigned version=cert::read_count(in),nscales=cert::read_count(in),nterms;
        cert::require(magic=="TRANSPORT_POLYNOMIAL"&&version==1&&nscales==2,"transport format mismatch");
        for(D &s:scales) s=convert(read_rational(in));
        cert::require(scales[0].lo>0&&scales[0].hi<scales[1].lo,"scales not strictly ordered");
        for(D &m:multipliers) m=convert(read_rational(in));
        nterms=cert::read_count(in);
        for(unsigned j=0;j<nterms;j++) {
            StorageTerm term; std::string kind;
            in>>kind;term.power0=cert::read_count(in);term.power1=cert::read_count(in);
            cert::require(term.power0<=unsigned(std::numeric_limits<int>::max()/2)
                          &&term.power1<=unsigned(std::numeric_limits<int>::max()/2)-term.power0,
                          "state degree exceeds integer arithmetic capacity");
            cert::require(kind=="h"||kind=="u","unknown storage term");
            term.moment=kind=="h";
            cert::require(term.moment?term.power0+term.power1==0:term.power0+term.power1>0,
                          "invalid monomial degree");
            term.coefficient=convert(read_rational(in));
            std::array<std::array<cert::ExactPolynomial,4>,2> exact;
            for(unsigned side=0;side<2;side++)for(unsigned index=0;index<4;index++) {
                auto &poly=term.quotients[side][index];
                unsigned count=cert::read_count(in);cert::require(count>0,"empty canceled polynomial");
                exact[side][index].resize(count);
                for(unsigned k=0;k<count;k++)poly.push_back(convert(read_rational(in,&exact[side][index][k])));
                cert::strip(exact[side][index]);
            }
            cert::verify_cancellation(exact,term.moment,term.power0+term.power1);
            terms.push_back(std::move(term));
        }
        cert::require(bool(in),"truncated transport polynomial");
        cert::require_end(in);
    }
};

struct Profile {
    A t,q,z,drift,ell;
    std::array<A,2> right,left;
    A right_linear,left_moment;
    D k;
    unsigned side;

    Profile()=default;

    Profile(cert::Reference &ref,const I &time,unsigned half) : k(convert(ref.par.k)),side(half) {
        cert::require(half<=1&&ref.filters.size()==2&&bool(ref.linear_filter),"incomplete profile flows");
        constexpr unsigned order=2;
        cert::Shape sh=ref.local_shape(time,order,true);
        t=boxes::variable(convert(time),2);
        q=half?A(2)-boxes::pow(t,3):boxes::pow(t,3);
        z=profile_jet(sh.z);
        drift=profile_jet(cert::inverse(sh.P)*I(3));
        ell=t*(A(2)-boxes::pow(t,3))/(boxes::square(z)*k);
        Jet small=cert::cube(cert::variable(time,order)),large=Jet(order,I(2))-small;
        for(unsigned j=0;j<2;j++) {
            Jet quotient=ref.normalized_filter(time,order,int(j));
            Jet reflected_filter=cert::reflected(ref.filters[j]->jet(I(2)-time,order));
            Jet l=half?reflected_filter/large:quotient;
            left[j]=unit_value(profile_jet(l*(ref.par.k/ref.par.scales[j])));
            right[j]=profile_jet(half?quotient:reflected_filter/large);
        }
        Jet linear_quotient=ref.normalized_filter(time,order,-1);
        Jet reflected_linear=cert::reflected(ref.linear_filter->jet(I(2)-time,order));
        right_linear=profile_jet(half?linear_quotient:reflected_linear/large);
        left_moment=unit_value(profile_jet(half?(large-reflected_linear)/large:
                                               Jet(order,I(1))-linear_quotient));
        cert::require(z.v.lo>0&&drift.v.lo>0,"profile positivity not enclosed");
    }
};

inline Profile centered_profile(const Profile &whole,const Profile &middle) {
    Profile result=whole;D delta=whole.t.v-middle.t.v;
    cert::require(middle.t.v.lo==middle.t.v.hi&&whole.t.v.lo<=middle.t.v.lo
                   &&middle.t.v.hi<=whole.t.v.hi,"invalid spatial Taylor center");
    for(auto pair:{std::pair{&result.q,&middle.q},std::pair{&result.z,&middle.z},
                   std::pair{&result.drift,&middle.drift},std::pair{&result.ell,&middle.ell},
                   std::pair{&result.right_linear,&middle.right_linear},
                   std::pair{&result.left_moment,&middle.left_moment}})
        centered_spatial_jet(*pair.first,*pair.second,delta);
    for(unsigned j=0;j<2;j++) {
        centered_spatial_jet(result.left[j],middle.left[j],delta);
        centered_spatial_jet(result.right[j],middle.right[j],delta);
    }
    return result;
}

struct Geometry { A U,V,mlo,mhi; };
inline Geometry geometry(D k,const std::array<D,2> &s,D ur,D tr) {
    A U=boxes::variable(ur,0),T=boxes::variable(tr,1),one(1);
    D c=k+s[0],a=k+s[1],delta=s[1]-s[0];
    A den=A(a)-U*delta;
    den.v=boxes::positive_floor(den.v,c);
    A V=unit_value(U*(A(c)+T*(one-U)*delta)/den);
    A node_p=U*(one-T)*a/(A(a)-U*T*delta);
    A node_m=(U*c+T*(one-U)*a)/(A(c)+T*(one-U)*delta);
    A lo=unit_value((one-U)*k/(A(k)+node_p*s[0]));
    A hi=unit_value(one-U*c/(A(k)+node_m*s[0]));
    return {U,V,lo,hi};
}

struct Contribution { A b,c; };
inline Contribution raw_storage(const StorageTerm &term,const Profile &p,
                                const std::array<D,2> &scales,
                                const A &U,const A &V,const A &m) {
    std::array<A,4> poly;
    for(unsigned j=0;j<4;j++) poly[j]=boxes::polynomial(term.quotients[p.side][j],p.q);
    A t2=boxes::square(p.t),z3=boxes::pow(p.z,3);
    if(term.moment) {
        return {-t2*poly[0]*m*p.k/p.z,
                (t2*poly[0]*p.drift*m+poly[1]*(p.q-m*D(2)))/z3};
    }
    unsigned total=term.power0+term.power1;
    A x0=U/p.k,x1=V/p.k;
    A W=boxes::pow(x0,term.power0)*boxes::pow(x1,term.power1);
    A first(0),quadratic(0);
    if(term.power0) {
        A derivative=boxes::pow(x0,term.power0-1)*boxes::pow(x1,term.power1)*D(int(term.power0));
        first=first+derivative;
        quadratic=quadratic+derivative*boxes::square(x0)*scales[0];
    }
    if(term.power1) {
        A derivative=boxes::pow(x0,term.power0)*boxes::pow(x1,term.power1-1)*D(int(term.power1));
        first=first+derivative;
        quadratic=quadratic+derivative*boxes::square(x1)*scales[1];
    }
    A b=-t2*(poly[0]*W*p.k+poly[1]*first)/p.z;
    A c=(t2*(poly[0]*p.drift*W+poly[1]*first*boxes::square(p.z))
          -poly[2]*W*D(int(2*total))-poly[3]*quadratic)/z3;
    return {b,c};
}

struct StateFormula {
    A b,c,ell;
    static StateFormula build(const Parameters &par,const Profile &p,D ur,D tr,unsigned moment_side) {
        Geometry g=geometry(p.k,par.scales,ur,tr);
        A m=moment_side?g.mhi:g.mlo;
        A common=(A(2)-boxes::pow(p.t,3))/(p.drift*p.z);
        A b=common*par.multipliers[2],c(0);
        std::array<A,2> states{g.U,g.V};
        for(unsigned j=0;j<2;j++) {
            A denominator=A(2)+p.q*states[j]*(par.scales[j]/p.k)
                           +(A(2)-p.q)*p.right[j];
            A coefficient=-common*(A(1)+states[j]*p.right[j])*par.scales[j]/denominator;
            b=b+coefficient*par.multipliers[j];
        }
        A a=A(1)-m;
        A second=common*(a*p.right_linear*D(2)-p.q*a-(A(2)-p.q)*p.right_linear)*(p.k/D(2));
        b=b+second*par.multipliers[3];
        A gauge_factor=boxes::square(p.z)*p.k/p.drift;
        for(const StorageTerm &term:par.terms) {
            Contribution state=raw_storage(term,p,par.scales,g.U,g.V,m);
            Contribution ref=raw_storage(term,p,par.scales,p.left[0],p.left[1],p.left_moment);
            b=b+(state.b+gauge_factor*ref.c)*term.coefficient;
            c=c+(state.c-ref.c)*term.coefficient;
        }
        return {b*par.dilation,c*par.dilation,p.ell};
    }
    A residual(const A &price) const {
        return boxes::cubic_conjugate(b-price)-c-price*(A(1)-ell);
    }
};

inline A determinant_ratio(const Profile &p,D scale,const A &right,const A &state) {
    // Exact cancellation at the opposite endpoint: if q=2 and R=s/k,
    // (1+U*R)/(2+q*(s/k)*U+(2-q)*R)=1/2 for every U.
    // The centered identity also encloses all derivatives and avoids
    // artificial state dependence as q approaches 2.
    // Proof: the analytical justification, Lemma lem:mobius.
    D a=scale/p.k;
    A delta=p.side?boxes::pow(p.t,3):A(2)-boxes::pow(p.t,3);
    A denominator=A(2)+p.q*state*a+delta*right;
    denominator.v=boxes::positive_floor(denominator.v,D(2));
    A direct=(A(1)+state*right)/denominator;
    A centered=A(D(1)/D(2))
        +(state*(right-A(a))+delta*(state*a-right)/D(2))/denominator;
    return boxes::intersect(direct,centered);
}

inline A second_moment_comparison(const Profile &p,const A &state) {
    // At q=2, the reflected normalized linear filter is exactly one.
    // Factor the common state before interval evaluation; otherwise
    // 2*state*R-q*state creates artificial uncertainty at this endpoint.
    // Proof: the analytical justification, Lemma lem:mobius.
    A delta=p.side?boxes::pow(p.t,3):A(2)-boxes::pow(p.t,3);
    A direct=state*p.right_linear*D(2)-p.q*state-delta*p.right_linear;
    A centered=state*(p.right_linear-A(1))*D(2)+delta*(state-p.right_linear);
    return boxes::intersect(direct,centered);
}

/* Polynomial/profile factors and reference centering are evaluated once
 * per spatial cell. State subdivision only evaluates the remaining low
 * degree monomials and two rational determinant coefficients.
 */
struct Monomials { A W,V,S; };
inline Monomials monomials(unsigned i,unsigned j,const A &U,const A &V,D k,
                           const std::array<D,2> &scales) {
    A x=U/k,y=V/k;
    Monomials out{boxes::pow(x,i)*boxes::pow(y,j),A(0),A(0)};
    if(i) {
        A derivative=boxes::pow(x,i-1)*boxes::pow(y,j)*D(int(i));
        out.V=out.V+derivative; out.S=out.S+derivative*boxes::square(x)*scales[0];
    }
    if(j) {
        A derivative=boxes::pow(x,i)*boxes::pow(y,j-1)*D(int(j));
        out.V=out.V+derivative; out.S=out.S+derivative*boxes::square(y)*scales[1];
    }
    return out;
}
class StateKernel {
    const Parameters &par;
    const Profile &p;
    struct TermData {
        bool moment;
        unsigned i,j;
        A bW,bV,cW,cV,cS,reference_b;
        Monomials reference;
    };
    std::vector<TermData> terms;
    A common;
public:
    StateKernel(const Parameters &parameters,const Profile &profile):par(parameters),p(profile),
        common((A(2)-boxes::pow(p.t,3))/(p.drift*p.z)) {
        A t2=boxes::square(p.t),z3=boxes::pow(p.z,3);
        A gauge=boxes::square(p.z)*p.k/p.drift;
        for(const auto &term:par.terms) {
            std::array<A,4> poly;
            for(unsigned j=0;j<4;j++)
                poly[j]=boxes::polynomial(term.quotients[p.side][j],p.q)*term.coefficient;
            TermData data;
            data.moment=term.moment; data.i=term.power0; data.j=term.power1;
            data.bW=-t2*poly[0]*p.k/p.z;
            if(term.moment) {
                data.cW=(t2*poly[0]*p.drift-poly[1]*D(2))/z3;
                data.reference_b=gauge*(data.cW*p.left_moment+poly[1]*p.q/z3);
            } else {
                data.bV=-t2*poly[1]/p.z;
                data.cW=(t2*poly[0]*p.drift-poly[2]*D(int(2*(data.i+data.j))))/z3;
                data.cV=t2*poly[1]/p.z;
                data.cS=-poly[3]/z3;
                data.reference=monomials(data.i,data.j,p.left[0],p.left[1],p.k,par.scales);
                data.reference_b=gauge*(data.cW*data.reference.W+data.cV*data.reference.V
                                        +data.cS*data.reference.S);
            }
            terms.push_back(std::move(data));
        }
    }
    void tighten_spatial(const StateKernel &middle,D delta) {
        cert::require(terms.size()==middle.terms.size(),"inconsistent spatial kernels");
        centered_spatial_jet(common,middle.common,delta);
        for(std::size_t j=0;j<terms.size();j++) {
            auto &a=terms[j];const auto &b=middle.terms[j];
            for(auto pair:{std::pair{&a.bW,&b.bW},std::pair{&a.bV,&b.bV},
                           std::pair{&a.cW,&b.cW},std::pair{&a.cV,&b.cV},
                           std::pair{&a.cS,&b.cS},std::pair{&a.reference_b,&b.reference_b},
                           std::pair{&a.reference.W,&b.reference.W},
                           std::pair{&a.reference.V,&b.reference.V},
                           std::pair{&a.reference.S,&b.reference.S}})
                centered_spatial_jet(*pair.first,*pair.second,delta);
        }
    }
    StateFormula evaluate(D ur,D tr,unsigned moment_side) const {
        Geometry g=geometry(p.k,par.scales,ur,tr);
        A m=moment_side?g.mhi:g.mlo;
        A b=common*par.multipliers[2],c(0);
        std::array<A,2> states{g.U,g.V};
        for(unsigned j=0;j<2;j++) {
            A ratio=determinant_ratio(p,par.scales[j],p.right[j],states[j]);
            b=b-common*ratio*(par.scales[j]*par.multipliers[j]);
        }
        A a=A(1)-m;
        b=b+common*second_moment_comparison(p,a)*(p.k/D(2)*par.multipliers[3]);
        for(const auto &term:terms) {
            if(term.moment) {
                b=b+term.bW*m+term.reference_b;
                c=c+term.cW*(m-p.left_moment);
            } else {
                Monomials state=monomials(term.i,term.j,g.U,g.V,p.k,par.scales);
                b=b+term.bW*state.W+term.bV*state.V+term.reference_b;
                c=c+term.cW*(state.W-term.reference.W)+term.cV*(state.V-term.reference.V)
                   +term.cS*(state.S-term.reference.S);
            }
        }
        return {b*par.dilation,c*par.dilation,p.ell};
    }
};
} // namespace transport
