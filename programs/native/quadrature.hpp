#pragma once
// Integrate a centered Taylor polynomial and enclose its integral remainder.
// Lemma lem:quadrature proves the bound 2*M_N*h^(N+1)/(N+1).
// Local width <= tolerance*cell_length gives the summed global width budget;
// parameter uncertainty is retained when a caller instead fixes a partition.
#include "trace.hpp"
#include <functional>

namespace cert {
struct QuadratureCell { double left,right; I value; };
using Integrand=std::function<Jet(const I &,unsigned)>;
struct Integral {
    I value;
    std::vector<QuadratureCell> cells;
};

inline Integral quadrature(Integrand f,double begin,double end,unsigned degree,I tol, const char *caller=__builtin_FILE(), unsigned line=__builtin_LINE()) {
    require(degree>=2&&tol.positive()&&begin<end,"invalid quadrature request");
    auto &trace=numerical_trace();
    const char *basename=std::strrchr(caller,'/');
    trace<<"quadrature "<<(basename?basename+1:caller)<<":"<<line
         <<" "<<begin<<" "<<end<<" degree "<<degree<<" tolerance "<<tol<<"\n";
    Integral result;
    std::vector<std::pair<double,double>> pending{{begin,end}};
    while(!pending.empty()) {
        auto [left,right]=pending.back(); pending.pop_back();
        I a(left),b(right),m=(a+b)/I(2),h=(b-a)/I(2);
        Jet center=f(m,degree-1),box=f(hull(a,b),degree);
        I val;
        for(unsigned j=0;j<degree;j+=2)
            val=val+I(2)*center[j]*pow(h,j+1)/I(int(j+1));
        I remainder=I(2)*abs_bound(box[degree])*pow(h,degree+1)/I(int(degree+1));
        val=val+centered(remainder);
        trace_cell("cell",left,right,center,box,val);
        if(mpfr_cmp(width(val).hi.x,(tol*(b-a)).lo.x)<=0) {
            result.value=result.value+val;
            trace<<"accept sum "<<result.value<<"\n";
            result.cells.push_back({left,right,val});
        } else {
            trace<<"subdivide\n";
            double mid=left+(right-left)/2;
            require(mid>left&&mid<right,"quadrature time format exhausted");
            pending.emplace_back(mid,right); pending.emplace_back(left,mid);
        }
    }
    trace<<"integral "<<result.value<<"\n"<<std::flush;
    require(bool(trace),"cannot save quadrature trace");
    return result;
}
} // namespace cert
