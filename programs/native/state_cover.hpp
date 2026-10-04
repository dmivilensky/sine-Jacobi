#pragma once
// Bound both conditional moment branches on a complete dyadic cover.
// Lemma lem:box-bound gives value/gradient/Hessian and concavity-based
// upper bounds. A sampled lower bound chooses work only; strict negative upper
// bounds on every leaf establish the continuous inequality. All ties in
// subdivision priority have an explicit geometric ordering.
#include "transport_kernel.hpp"
#include <queue>
#include <string>
#include <iomanip>

namespace transport {

inline double cell_center(double left,double right) {
    // This exact binary64 point, rather than the possibly unrepresentable
    // real arithmetic mean, defines the affine price. Taylor's integral
    // identity only needs a center in the cell; see Lemma lem:box-bound.
    // Each operation uses the checked round-to-nearest environment.
    double difference=right-left;
    double half_difference=difference/2;
    double center=left+half_difference;
    cert::require(center>left&&center<right,"spatial center format exhausted");
    return center;
}

inline double negative_margin(double upper) {
    cert::require(std::isfinite(upper)&&upper<0,"a strict finite margin is required");
    int exponent;std::frexp(-upper,&exponent);
    // Rounding a negative upper bound toward zero to a power of two gives
    // a shorter exact certificate, with an explicit strict dyadic margin.
    double margin=std::ldexp(-1.,exponent-1);
    cert::require(margin>=upper&&margin<0,"dyadic margin rounding failed");
    return margin;
}

struct Price {
    double center=0,intercept=0,slope=0;
    A at(D t) const {
        return A(D(intercept))+(boxes::variable(t,2)-A(D(center)))*D(slope);
    }
};
struct Rectangle {
    D u=D(0.,1.),v=D(0.,1.);
    unsigned branch=0;
    std::string path;
};
struct Bound {
    Rectangle box;
    double upper=INFINITY,lower=-INFINITY;
    std::array<double,2> point{},score{};
    A point_value;
    bool operator<(const Bound &b) const {
        // A total tie rule makes equal bounds independent of heap layout.
        if(upper!=b.upper)return upper<b.upper;
        if(box.branch!=b.box.branch)return box.branch>b.box.branch;
        return box.path>b.box.path;
    }
};

/* The only acceptance test is a rigorously rounded upper bound <= 0.
 * The midpoint/branch heuristics affect work and certificate size only.
 * Taylor's integral remainder needs only C^2 regularity, including the
 * matching point of the positive cubic in the conjugate.
 */
class Cover {
    Profile whole;
    const Profile &middle;
    StateKernel kernel,point_kernel;
    D times;
public:
    std::size_t evaluations=0;
    Cover(const Parameters &par,const Profile &p,const Profile &m)
        :whole(centered_profile(p,m)),middle(m),kernel(par,whole),point_kernel(par,m),times(p.t.v){
        kernel.tighten_spatial(point_kernel,times-middle.t.v);
    }

    Bound bound(const Rectangle &original,const Price &price) {
        Bound result; result.box=original;
        D u=original.u,v=original.v;
        A f=kernel.evaluate(u,v,original.branch).residual(price.at(times));
        ++evaluations;
        // Derivative signs on the whole rectangle allow exact boundary
        // reduction, uniformly for every spatial coordinate in this cell.
        if(f.g[0].lo>=0) u=D(u.hi);
        else if(f.g[0].hi<=0) u=D(u.lo);
        if(f.g[1].lo>=0) v=D(v.hi);
        else if(f.g[1].hi<=0) v=D(v.lo);
        if(u.lo!=original.u.lo||u.hi!=original.u.hi||v.lo!=original.v.lo||v.hi!=original.v.hi) {
            f=kernel.evaluate(u,v,original.branch).residual(price.at(times)); ++evaluations;
        }
        double cu=u.lo+(u.hi-u.lo)/2,cv=v.lo+(v.hi-v.lo)/2;
        A center=kernel.evaluate(D(cu),D(cv),original.branch).residual(price.at(times));
        A mid=point_kernel.evaluate(D(cu),D(cv),original.branch)
                              .residual(price.at(middle.t.v));
        evaluations+=2;
        result.point={cu,cv}; result.point_value=mid;
        result.lower=mid.v.lo;
        D du=u-D(cu),dv=v-D(cv),dt=times-middle.t.v;
        D base=mid.v+mid.g[2]*dt+D(std::max(0.,center.h[2][2].hi))*boxes::square(dt)/D(2);
        double base_upper=std::min(base.hi,center.v.hi);
        D taylor=D(base_upper)+center.g[0]*du+center.g[1]*dv
                 +D(f.h[0][0].hi)*boxes::square(du)/D(2)
                 +f.h[0][1]*du*dv+D(f.h[1][1].hi)*boxes::square(dv)/D(2);
        D first=D(base_upper)+f.g[0]*du+f.g[1]*dv;
        result.upper=std::min({f.v.hi,taylor.hi,first.hi});
        // If H <= -M with M positive definite, maximize the resulting
        // quadratic over all R^2. Dropping the rectangle restriction can
        // only enlarge this upper bound. The cross term uses |H_12|.
        D aa(-f.h[0][0].hi),bb(-f.h[1][1].hi),cc(boxes::magnitude(f.h[0][1]));
        D gu(boxes::magnitude(center.g[0])),gv(boxes::magnitude(center.g[1]));
        if(u.lo==u.hi) {
            if(bb.lo>0) result.upper=std::min(result.upper,(D(base_upper)+boxes::square(gv)/(D(2)*bb)).hi);
        } else if(v.lo==v.hi) {
            if(aa.lo>0) result.upper=std::min(result.upper,(D(base_upper)+boxes::square(gu)/(D(2)*aa)).hi);
        } else if(aa.lo>0&&bb.lo>0) {
            D determinant=aa*bb-boxes::square(cc);
            if(determinant.lo>0) {
                D excess=(bb*boxes::square(gu)+aa*boxes::square(gv)+D(2)*cc*gu*gv)
                         /(D(2)*determinant);
                result.upper=std::min(result.upper,(D(base_upper)+excess).hi);
            }
        }
        if(u.lo==u.hi&&v.lo==v.hi) result.upper=std::min(result.upper,base_upper);
        double wu=u.hi-u.lo,wv=v.hi-v.lo;
        result.score={wu*(boxes::magnitude(center.g[0])+wu*boxes::magnitude(f.h[0][0])
                          +wv*boxes::magnitude(f.h[0][1])),
                      wv*(boxes::magnitude(center.g[1])+wv*boxes::magnitude(f.h[1][1])
                          +wu*boxes::magnitude(f.h[0][1]))};
        cert::require(std::isfinite(result.upper)&&std::isfinite(result.lower),"nonfinite cover bound");
        // Retain the interval inputs to each mathematical upper bound,
        // including rejected boxes and the monotonicity reduction.
        auto &trace=cert::numerical_trace();
        trace<<"state-bound "<<times.lo<<" "<<times.hi<<" "<<original.branch<<" "
             <<(original.path.empty()?"-":original.path)<<" price "<<price.center<<" "
             <<price.intercept<<" "<<price.slope<<" box "<<u.lo<<" "<<u.hi<<" "<<v.lo<<" "<<v.hi;
        for(D a:{f.v,f.g[0],f.g[1],center.v,center.g[0],center.g[1],
                 f.h[0][0],f.h[0][1],f.h[1][1],mid.v,mid.g[2],center.h[2][2]})
            trace<<" "<<a.lo<<" "<<a.hi;
        trace<<" bounds "<<result.lower<<" "<<result.upper<<"\n";
        return result;
    }

    struct Maximum {
        double lower=-INFINITY,upper=INFINITY;
        Bound active;
        std::vector<Bound> leaves;
        bool accepted=false,exhausted=false,budget_exhausted=false;
    };

    Maximum maximum(const Price &price,double tolerance,std::size_t budget,bool certify=false,
                    const std::vector<Rectangle> &initial={}) {
        cert::require(tolerance>0&&budget>=2,"invalid state search resources");
        Maximum out;
        std::priority_queue<Bound> heap;
        auto push=[&](const Rectangle &r) {
            Bound b=bound(r,price);
            if(b.lower>out.lower) { out.lower=b.lower; out.active=b; }
            heap.push(std::move(b));
        };
        // A seed is a previously complete partition. Its geometry is
        // unchanged by the new spatial cell, price, or homothetic scale;
        // every bound is recomputed before it can be accepted here.
        if(initial.empty()) {
            push(Rectangle{D(0.,1.),D(0.,1.),0,""});
            push(Rectangle{D(0.,1.),D(0.,1.),1,""});
        } else for(const auto &rectangle:initial)push(rectangle);
        std::size_t splits=heap.size()>2?heap.size()-2:0;
        for(;;) {
            out.upper=heap.top().upper;
            if(certify&&out.upper<0) { out.accepted=true; break; }
            if(!certify&&out.upper-out.lower<=tolerance) break;
            if(certify&&out.lower>0) break;
            if(splits>=budget) {out.exhausted=true;out.budget_exhausted=true;break;}
            // When both state directions have been eliminated by a
            // monotonicity proof, only spatial/rounding uncertainty
            // remains. Further subdivision of the original state box
            // cannot reduce that uncertainty.
            if(heap.top().score[0]==0&&heap.top().score[1]==0) {
                out.exhausted=true;break;
            }
            Bound b=heap.top();heap.pop();
            unsigned axis=b.score[0]>=b.score[1]?0:1;
            Rectangle left=b.box,right=b.box;
            D range=axis?b.box.v:b.box.u;
            double midpoint=range.lo+(range.hi-range.lo)/2;
            if(!(midpoint>range.lo&&midpoint<range.hi)) {
                heap.push(std::move(b));out.exhausted=true;break;
            }
            if(axis) {left.v.hi=midpoint;right.v.lo=midpoint;}
            else {left.u.hi=midpoint;right.u.lo=midpoint;}
            left.path+=axis?'2':'0';right.path+=axis?'3':'1';
            push(left);push(right);++splits;
        }
        while(!heap.empty()) {out.leaves.push_back(heap.top());heap.pop();}
        cert::numerical_trace()<<"state-maximum "<<out.lower<<" "<<out.upper
                              <<" leaves "<<out.leaves.size()<<"\n"<<std::flush;
        return out;
    }
};

inline double price_derivative(const StateFormula &state,double e) {
    // A proposal, never an enclosure used for acceptance.
    double z=(state.b.v.lo+state.b.v.hi)/2-e;
    return std::pow(std::max(0.,1+2*z/3),2)-(state.ell.v.lo+state.ell.v.hi)/2;
}

inline Rectangle decode_rectangle(unsigned branch,const std::string &path) {
    cert::require(branch<=1,"invalid conditional moment branch");
    Rectangle r{D(0.,1.),D(0.,1.),branch,path};
    for(char c:path) {
        cert::require(c>='0'&&c<='3',"invalid subdivision instruction");
        D &range=c<'2'?r.u:r.v;
        double mid=range.lo+(range.hi-range.lo)/2;
        cert::require(mid>range.lo&&mid<range.hi,"state coordinate exhausted");
        if(c=='0'||c=='2') range.hi=mid;else range.lo=mid;
    }
    return r;
}
} // namespace transport
