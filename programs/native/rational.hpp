#pragma once
#include "interval.hpp"
#include <regex>
#include <array>

namespace cert {
// Exact Q arithmetic for the polynomial identities in lem:regular-criterion.
// GMP manual: https://gmplib.org/manual/Rational-Number-Functions
// Check these identities before numerical state evaluation.
struct Rational {
    mpq_t value;
    Rational() { mpq_init(value); }
    explicit Rational(long n):Rational() { mpq_set_si(value,n,1); }
    explicit Rational(const std::string &token):Rational() {
        static const std::regex syntax("-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?");
        require(std::regex_match(token,syntax),"exact integer/fraction coefficient required");
        require(mpq_set_str(value,token.c_str(),10)==0,"invalid rational coefficient");
        mpq_canonicalize(value);
    }
    Rational(const Rational &a):Rational() { mpq_set(value,a.value); }
    Rational(Rational &&a) noexcept:Rational() { mpq_swap(value,a.value); }
    Rational &operator=(const Rational &a) { if(this!=&a)mpq_set(value,a.value);return *this; }
    Rational &operator=(Rational &&a) noexcept { mpq_swap(value,a.value);return *this; }
    ~Rational() { mpq_clear(value); }
    bool zero() const { return mpq_sgn(value)==0; }
};
inline bool operator==(const Rational &a,const Rational &b) { return mpq_equal(a.value,b.value)!=0; }
inline Rational operator+(const Rational &a,const Rational &b) {
    Rational r;mpq_add(r.value,a.value,b.value);return r;
}
inline Rational operator-(const Rational &a,const Rational &b) {
    Rational r;mpq_sub(r.value,a.value,b.value);return r;
}
inline Rational operator*(const Rational &a,const Rational &b) {
    Rational r;mpq_mul(r.value,a.value,b.value);return r;
}
using ExactPolynomial=std::vector<Rational>;
inline void strip(ExactPolynomial &p) { while(p.size()>1&&p.back().zero())p.pop_back(); }
inline ExactPolynomial times_coordinate(const ExactPolynomial &p) {
    ExactPolynomial out(1);out.insert(out.end(),p.begin(),p.end());strip(out);return out;
}
inline ExactPolynomial times_wall(const ExactPolynomial &p) {
    ExactPolynomial out(p.size()+1);
    for(std::size_t j=0;j<p.size();j++) {
        out[j]=out[j]+Rational(2)*p[j];out[j+1]=out[j+1]-p[j];
    }
    strip(out);return out;
}
inline void verify_cancellation(const std::array<std::array<ExactPolynomial,4>,2> &q,
                                bool moment,unsigned degree) {
    for(unsigned j=0;j<4;j++)require(times_coordinate(q[0][j])==times_wall(q[1][j]),
                                    "left/right exact cancellation identity fails");
    ExactPolynomial expected;
    const auto &p=q[0][1];
    if(moment) {
        expected.assign(std::max(std::size_t(1),p.size()-1),Rational());
        for(std::size_t j=1;j<p.size();j++)expected[j-1]=Rational(long(j))*p[j];
        require(q[0][2]==ExactPolynomial(1)&&q[0][3]==ExactPolynomial(1),"nonzero unused moment polynomial");
    } else {
        // If L=q^(d-2)*p, then q^(d-1)*p'=(2-d)*L+q*L'.
        // For d=1 reconstruct p=q*L; for d>2 require divisibility by q^(d-2).
        for(std::size_t j=0;j<std::min(p.size(),std::size_t(degree-std::min(degree,2u)));j++)
            require(p[j].zero(),"state polynomial lacks its required zero at q=0");
        expected.resize(p.size());
        for(std::size_t j=0;j<p.size();j++)expected[j]=Rational(2-long(degree)+long(j))*p[j];
        require(q[0][2]==times_coordinate(p)&&q[0][3]==times_coordinate(times_coordinate(p)),
                "exact state polynomial powers disagree");
    }
    strip(expected);
    require(q[0][0]==expected,"exact storage derivative identity fails");
}
} // namespace cert
