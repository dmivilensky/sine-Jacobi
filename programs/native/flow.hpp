#pragma once
// Validated Taylor integration of scalar Riccati equations.
// General framework: Nedialkov, Jackson and Corliss, Applied Mathematics
// and Computation 105 (1999), 21--68, DOI 10.1016/S0096-3003(98)10083-8.
// The scalar Picard and one-sided stability proof is lem:flow-step;
// the removable endpoint quotient is lem:cubic-quotient.
#include "trace.hpp"
#include <functional>
#include <fstream>
#include <limits>

namespace cert {
/* Scalar quadratic ODE y'=C(t)+A(t)y+B(t)y^2.
 * Input provider encloses all normalized derivatives of C,A,B on a point
 * or interval. A Picard tube validates every intermediate state. Taylor's
 * integral remainder validates the truncation. Initial uncertainty is
 * propagated by exp(h*sup(A+2By)); a negative one-sided derivative never
 * causes the spurious exponential growth of naive interval propagation.
 */
struct QuadraticODE { Jet C,A,B; };
using Provider = std::function<QuadraticODE(const I &,unsigned)>;

inline Jet ode_jet(const QuadraticODE &f,const I &y,unsigned degree) {
    // Shift y=y0+z before coefficient convolution. In particular the two
    // occurrences of B*y0*z are combined with A*z, preserving the negative
    // derivative of a stable Riccati equation inside interval arithmetic.
    Jet r(degree),C=f.C+f.A*y+f.B*square(y),A=f.A+f.B*(I(2)*y);
    std::vector<I> squares(degree);
    for(unsigned n=0;n<degree;n++) {
        // r0=0 until the final assignment. Compute [r^2]_n once; the
        // shifted Riccati recurrence then costs O(degree^2) operations.
        for(unsigned j=1;j<(n+1)/2;j++)squares[n]=squares[n]+I(2)*r[j]*r[n-j];
        if(n>0&&n%2==0)squares[n]=squares[n]+square(r[n/2]);
        I value=C[n];
        for(unsigned j=0;j<=n;j++) {
            value=value+A[j]*r[n-j];
            value=value+f.B[j]*squares[n-j];
        }
        r[n+1]=value/I(int(n+1));
    }
    r[0]=y;
    return r;
}

struct FlowCell {
    double left,right;
    Jet polynomial;
    I uncertainty;    // uniform radius over the entire cell
    I initial_radius;
    I remainder_coefficient;
    I tube;
    I lipschitz_upper;
    I range(const I &t) const {
        require(contained(t,hull(I(left),I(right))),"query outside flow cell");
        I dt=t-I(left);
        I current_radius=upper(remainder_coefficient*pow(upper(dt),polynomial.order()+1)
                               +initial_radius*exp(lipschitz_upper*dt));
        return intersect(evaluate(polynomial,dt,polynomial.order())+
                         centered(current_radius),tube);
    }
};

class Flow {
public:
    Provider forcing;
    unsigned order;
    I tolerance;
    std::vector<FlowCell> cells;
    I endpoint;
    std::size_t rejected=0;

    Flow(Provider f,unsigned p,I tol) : forcing(std::move(f)),order(p),tolerance(tol) {
        require(order>=2&&tolerance.positive(),"invalid flow accuracy request");
    }

    double segment(double begin,double end,I initial,double initial_step=1.,
                   std::function<bool(double,const I &)> stop={}) {
        require(begin<end&&initial_step>0,"invalid flow segment");
        numerical_trace()<<"flow-begin "<<begin<<" "<<end<<" order "<<order
                         <<" tolerance "<<tolerance<<" initial "<<initial<<"\n";
        if(!cells.empty()) require(cells.back().right==begin,"disconnected flow segments");
        double t=begin,h=std::min(initial_step,end-begin);
        while(t<end) {
            double next=std::min(end,t+h);
            require(next>t,"flow refinement exhausted the time format");
            I step=I(next)-I(t),time=hull(I(t),I(next));
            I m=midpoint(initial),initial_radius=radius(initial,m);
            QuadraticODE box=forcing(time,order);
            auto rhs=[&](const I &y) { return box.C[0]+box.A[0]*y+box.B[0]*square(y); };
            I extension=abs_bound(rhs(initial))*step+initial_radius;
            // A positive dyadic padding avoids equality at a zero solution.
            I pad=tolerance*step;
            extension=upper(extension+pad);
            I tube;
            bool enclosed=false;
            for(unsigned attempt=0;attempt<=order;attempt++) {
                tube=m+centered(extension*I(2));
                I image=hull(initial,m)+hull(I(0),step)*rhs(tube);
                numerical_trace()<<"Picard "<<t<<" "<<next<<" attempt "<<attempt
                                 <<" tube "<<tube<<" image "<<image<<"\n";
                if(contained(image,tube)) { enclosed=true; break; }
                extension=abs_bound(image-m)+extension;
            }
            if(!enclosed) { ++rejected; h/=2; continue; }
            Jet remainder=ode_jet(box,tube,order+1);
            I tail=abs_bound(remainder[order+1])*pow(step,order+1);
            numerical_trace()<<"flow-tail "<<t<<" "<<next<<" "<<tail<<"\n";
            if(mpfr_cmp(tail.hi.x,(tolerance*step).lo.x)>0) {
                ++rejected; h/=2; continue;
            }
            Jet polynomial=ode_jet(forcing(I(t),order-1),m,order);
            I lip=upper(box.A[0]+I(2)*box.B[0]*tube);
            // For a uniform-in-time tube use max(1,exp(h L)). At the next
            // node the sharper exp(h L) is valid even when L is negative.
            I factor=exp(step*lip);
            I uniform_factor=hull(I(1),factor);
            I uniform_radius=upper(tail+initial_radius*upper(uniform_factor));
            I endpoint_radius=upper(tail+initial_radius*factor);
            FlowCell cell{t,next,std::move(polynomial),uniform_radius,
                          initial_radius,abs_bound(remainder[order+1]),tube,lip};
            endpoint=intersect(evaluate(cell.polynomial,step,order)+
                               centered(endpoint_radius),tube);
            trace_cell("flow-derivatives",t,next,cell.polynomial,remainder,endpoint);
            numerical_trace()<<"flow-radius "<<initial_radius<<" lip "<<lip
                             <<" uniform "<<uniform_radius<<" terminal "<<endpoint_radius<<"\n";
            endpoint.check(); cells.push_back(std::move(cell));
            initial=endpoint; t=next;
            if((cells.size()&(cells.size()-1))==0) {
                std::cerr<<"flow cells "<<cells.size()<<" t "<<t
                         <<" step "<<h<<" radius "<<width(endpoint).upper_double()
                         <<" rejected "<<rejected<<"\n";
            }
            // Refinement choices affect cost only. Every accepted step has
            // already passed the complete tube and remainder predicates.
            if(mpfr_cmp((tail*I(2)).hi.x,(tolerance*step).lo.x)<0) h*=2;
            if(stop&&stop(t,endpoint)) return t;
        }
        return t;
    }

    I range(const I &t) const {
        require(!cells.empty(),"empty flow");
        require(t.lower_double()>=cells.front().left&&t.upper_double()<=cells.back().right,
                "flow query outside domain");
        auto first=std::lower_bound(cells.begin(),cells.end(),t.lower_double(),
             [](const FlowCell &c,double v){return c.right<v;});
        bool any=false; I result;
        for(auto it=first;it!=cells.end()&&it->left<=t.upper_double();++it) {
            I portion=intersect(t,hull(I(it->left),I(it->right)));
            I v=it->range(portion);
            result=any?hull(result,v):v; any=true;
        }
        require(any,"flow query found no covering cell");
        return result;
    }

    Jet jet(const I &t,unsigned degree) const {
        if(degree==0) return Jet(0,range(t));
        return ode_jet(forcing(t,degree-1),range(t),degree);
    }

    void save(const std::string &path) const {
        auto out=output_file(path);
        out<<"QUADRATIC_FLOW 1\n"<<precision<<" "<<order<<" "<<cells.size()<<"\n";
        out<<tolerance<<"\n";
        for(const auto &c:cells) {
            out<<std::hexfloat<<c.left<<" "<<c.right<<"\n";
            out<<c.initial_radius<<"\n"<<c.remainder_coefficient<<"\n";
            out<<c.tube<<"\n"<<c.lipschitz_upper<<"\n";
            for(const I &a:c.polynomial.c) out<<a<<"\n";
        }
        out<<endpoint<<"\n"; finish_output(out);
        numerical_trace()<<"flow-saved "<<path.substr(path.find_last_of('/')+1)<<" cells "<<cells.size()<<"\n"<<std::flush;
    }

private:
    void read_saved(const std::string &path,double begin,double end,I initial,bool verify_equation) {
        require(cells.empty(),"cannot load over a nonempty flow");
        std::ifstream in(path); require(bool(in),"cannot read saved flow");
        std::string magic; unsigned version=0,saved_order=0; long saved_precision=0;
        std::size_t count=0; I saved_tolerance;
        in>>magic;version=read_count(in);saved_precision=read_count(in);
        saved_order=read_count(in);count=read_count(in);in>>saved_tolerance;
        require(bool(in)&&magic=="QUADRATIC_FLOW"&&version==1&&saved_order==order,
                "saved flow format/order mismatch");
        require(saved_precision>=64&&saved_precision<=precision&&count>0&&saved_tolerance.positive(),
                "invalid saved precision, tolerance or empty flow");
        double previous=begin;
        for(std::size_t i=0;i<count;i++) {
            std::string ls,rs; in>>ls>>rs;
            std::size_t used_left=0,used_right=0;
            double left=std::stod(ls,&used_left),right=std::stod(rs,&used_right);
            require(used_left==ls.size()&&used_right==rs.size(),"invalid saved coordinate");
            require(std::isfinite(left)&&std::isfinite(right),"nonfinite saved coordinate");
            require(contained(I(ls),I(left))&&contained(I(rs),I(right)),
                    "saved coordinate is not exactly representable in binary64");
            require(left==previous&&left<right&&right<=end,"saved cells do not form a partition");
            I initial_radius,remainder_coefficient,tube,lip;
            in>>initial_radius>>remainder_coefficient>>tube>>lip;
            Jet polynomial(order);
            for(I &c:polynomial.c) in>>c;
            require(bool(in),"truncated saved flow");
            require(initial_radius.nonnegative()&&remainder_coefficient.nonnegative(),
                    "negative saved error radius");
            require(width(polynomial[0]).zero(),"Taylor expansion center must be an exact point");
            require(contained(initial,polynomial[0]+centered(initial_radius)),
                    "incoming initial interval not enclosed");
            I step=I(right)-I(left),time=hull(I(left),I(right));
            if(verify_equation) {
                QuadraticODE box=forcing(time,order);
                I rhs=box.C[0]+box.A[0]*tube+box.B[0]*square(tube);
                // The Taylor polynomial follows the solution starting at its
                // exact center, while stability compares that solution with
                // every possible incoming state. The tube must cover BOTH.
                // Containment in a radius alone does not put the center in
                // the incoming enclosure, especially after a tighter replay.
                I initial_family=hull(initial,polynomial[0]);
                require(contained(initial_family+hull(I(0),step)*rhs,tube),"saved Picard tube fails");
                Jet coefficients=ode_jet(forcing(I(left),order-1),polynomial[0],order);
                for(unsigned j=0;j<=order;j++)
                    require(contained(coefficients[j],polynomial[j]),"saved Taylor coefficient fails");
                Jet whole=ode_jet(box,tube,order+1);
                trace_cell("flow-verification",left,right,coefficients,whole,tube);
                require(mpfr_cmp(abs_bound(whole[order+1]).hi.x,remainder_coefficient.hi.x)<=0,
                        "saved remainder coefficient fails");
                I computed_lip=box.A[0]+I(2)*box.B[0]*tube;
                require(mpfr_cmp(computed_lip.hi.x,lip.lo.x)<=0,"saved one-sided derivative fails");
                numerical_trace()<<"verified-lip "<<computed_lip<<" saved "<<lip<<"\n";
            }
            I tail=remainder_coefficient*pow(step,order+1);
            I factor=exp(step*lip);
            I uniform=upper(tail+initial_radius*upper(hull(I(1),factor)));
            endpoint=intersect(evaluate(polynomial,step,order)+
                               centered(upper(tail+initial_radius*factor)),tube);
            cells.push_back({left,right,std::move(polynomial),uniform,initial_radius,
                             remainder_coefficient,tube,lip});
            initial=endpoint; previous=right;
        }
        I saved_endpoint; in>>saved_endpoint;
        require(bool(in)&&previous==end,"saved flow misses the terminal endpoint");
        require(contained(endpoint,saved_endpoint),"saved endpoint fails independent propagation");
        require_end(in);
        std::cerr<<(verify_equation?"independently rechecked ":"loaded unverified proposal cache: ")
                 <<cells.size()<<" saved flow cells\n";
    }
public:
    void load_and_verify(const std::string &path,double begin,double end,I initial) {
        read_saved(path,begin,end,initial,true);
    }
    void load_proposal(const std::string &path,double begin,double end,I initial) {
        // This path produces proposals only. It establishes no ODE
        // enclosure. Every acceptance reader uses load_and_verify.
        read_saved(path,begin,end,initial,false);
    }
};
/* Taylor model for an analytically removable cubic zero.
 * If f(0)=f'(0)=f''(0)=0 and f_j=f^(j)/j!, then for 0<=t<=b
 *
 *  (f/t^3)_j(t) in sum_{l=j}^N binom(l,j) f_{l+3}(0)t^(l-j)
 *                   +binom(N+1,j) f_{N+4}([0,b]) t^(N+1-j).
 *
 * Expand inside the positive integral representation of f/t^3 to obtain
 * this derivative remainder. In particular it remains valid at t=0.
 */
class CubicQuotient {
    unsigned degree;
    double cutoff;
    std::vector<I> coefficients;
    I remainder;
    static I choose(unsigned n,unsigned k) {
        I value(1);
        for(unsigned j=1;j<=k;j++) value=value*I(int(n-j+1))/I(int(j));
        return value;
    }
public:
    CubicQuotient(std::function<Jet(const I &,unsigned)> source,unsigned n,double b)
        :degree(n),cutoff(b) {
        require(cutoff>0&&degree>=2,"invalid cubic-quotient model");
        Jet origin=source(I(0),degree+3);
        for(unsigned j=0;j<3;j++)
            require(origin[j].zero(),"cubic quotient requires three exactly vanishing initial coefficients");
        for(unsigned j=0;j<=degree;j++) coefficients.push_back(origin[j+3]);
        remainder=source(hull(I(0),I(cutoff)),degree+4)[degree+4];
    }
    Jet evaluate(const I &t,unsigned n) const {
        require(n<=degree&&contained(t,hull(I(0),I(cutoff))),"quotient query outside model");
        Jet out(n);
        for(unsigned j=0;j<=n;j++) {
            I value=coefficients[degree]*choose(degree,j);
            for(unsigned l=degree;l>j;l--)
                value=value*t+coefficients[l-1]*choose(l-1,j);
            out[j]=value+remainder*choose(degree+1,j)*pow(t,degree+1-j);
        }
        return out;
    }
    unsigned order() const { return degree; }
};
} // namespace cert
