#pragma once
// Ground-exact polynomial family and a complete parameter-cube proof.
// The shape and mass formulas are thm:primal-map; reflected sensitivities
// and the exponential energy ceiling are prop:primal-jacobian. Acceptance
// uses ||CF(0)||+rho*||I-C*DF(cube)||<rho, with ||I-C*DF(cube)||<1,
// as proved in lem:contraction. General verification: Rump (2010),
// DOI 10.1017/S096249291000005X.
#include "flow.hpp"
#include "quadrature.hpp"
#include <filesystem>
#include <memory>
#include <set>

namespace cert {
struct PrimalParameters {
    std::string beta_token;
    bool second=false;
    I beta,k;
    std::vector<std::string> scale_tokens;
    std::vector<I> scales,coefficients;
    std::vector<unsigned> directions;
    std::vector<std::vector<I>> C;

    explicit PrimalParameters(const std::string &path) {
        std::ifstream input(path);require(bool(input),"cannot read primal profile");
        std::string magic,token; input>>magic;
        unsigned version=read_count(input);input>>beta_token;
        unsigned second_flag=read_count(input),count=read_count(input),dimension;
        require(magic=="GROUND_PRIMAL"&&version==1&&second_flag<=1,"invalid primal format");
        second=second_flag;beta=I(beta_token);k=square(pi())/I(4);
        require(beta.nonnegative(),"negative large-scale parameter");
        for(unsigned j=0;j<count;j++) {
            input>>token;scale_tokens.push_back(token);scales.emplace_back(token);
            require(scales.back().positive(),"nonpositive primal scale");
        }
        for(unsigned j=1;j<scales.size();j++)require((scales[j]-scales[j-1]).positive(),"unordered primal scales");
        if(beta.positive())scales.push_back(beta*k);
        count=read_count(input);require(count>0,"empty primal polynomial");
        for(unsigned j=0;j<count;j++) {input>>token;coefficients.emplace_back(token);}
        dimension=read_count(input);
        require(dimension==2+scales.size()+unsigned(second),"primal constraint count mismatch");
        for(unsigned j=0;j<dimension;j++) {
            unsigned index=read_count(input);require(index<coefficients.size(),"invalid correction degree");
            directions.push_back(index);
        }
        require(std::set<unsigned>(directions.begin(),directions.end()).size()==dimension,
                "repeated correction degree");
        C.assign(dimension,std::vector<I>(dimension));
        for(auto &row:C)for(I &entry:row){input>>token;entry=I(token);}
        require(bool(input),"truncated primal profile");
        require_end(input);
    }
    unsigned dimension() const {return directions.size();}
    void make_dyadic() {
        // These are new exact proposal points, not rounded proof bounds.
        // Start with the short native binary64 representation. The producer
        // may use longer MPFR dyadics if this representation stagnates.
        for(I &c:coefficients)c=I(midpoint(c).lower_double());
        for(auto &row:C)for(I &entry:row)entry=I(midpoint(entry).lower_double());
    }
    void save(const std::string &path) const {
        auto out=cert::output_file(path);require(bool(out),"cannot write primal profile");
        out<<"GROUND_PRIMAL 1\n"<<beta_token<<"\n"<<unsigned(second)<<"\n"<<scale_tokens.size()<<"\n";
        for(const auto &s:scale_tokens)out<<s<<"\n";
        out<<coefficients.size()<<"\n";
        for(const I &c:coefficients) {
            require(width(c).zero(),"central polynomial coefficient is not exact");
            out<<hex_real(c.lo.x)<<"\n";
        }
        out<<dimension()<<"\n";for(unsigned d:directions)out<<d<<"\n";
        for(const auto &row:C)for(const I &entry:row) {
            require(width(entry).zero(),"preconditioner coefficient is not exact");
            out<<hex_real(entry.lo.x)<<"\n";
        }
        finish_output(out);
    }
};

inline std::vector<Jet> chebyshev_basis(const Jet &x,unsigned degree) {
    std::vector<Jet> basis{Jet(x.order(),I(1))};
    if(degree)basis.push_back(x);
    for(unsigned j=1;j<degree;j++)basis.push_back(I(2)*x*basis[j]-basis[j-1]);
    return basis;
}

struct PrimalShape {Jet t,h,Q,a,n,P;};
inline PrimalShape primal_shape(const PrimalParameters &par,const I &point,unsigned degree) {
    Jet t=variable(point,degree),one(degree,I(1));
    auto basis=chebyshev_basis(t*I(2)-one,par.coefficients.size()-1);
    Jet p(degree);
    for(unsigned j=0;j<basis.size();j++)p=p+basis[j]*par.coefficients[j];
    Jet a=exp(p),h=one*I(2)-cube(t),Q=cube(t)*h;
    return {t,h,Q,a,t*sqrt((a+t*h)/par.k),one*I(3)/a};
}

struct PrimalEvaluation {
    PrimalParameters par;
    unsigned order;
    I flow_tolerance,quad_tolerance;
    std::vector<std::unique_ptr<Flow>> filters;
    std::vector<std::pair<double,double>> partition;
    std::vector<I> integrals;

    PrimalEvaluation(PrimalParameters parameters,unsigned n,I ft,I qt)
        :par(std::move(parameters)),order(n),flow_tolerance(ft),quad_tolerance(qt) {
        require(order>=2&&ft.positive()&&qt.positive(),"invalid primal accuracy request");
    }

    void build_flows(const std::string &directory,bool replay) {
        std::filesystem::create_directories(directory);
        for(unsigned j=0;j<par.scales.size()+unsigned(par.second);j++) {
            bool linear=j==par.scales.size();I scale=linear?par.k:par.scales[j];
            auto flow=std::make_unique<Flow>([&,scale,linear](const I &v,unsigned n) {
                // At the shared node, Taylor coefficients are those of
                // the segment starting there (the right half). A closed
                // nondegenerate left interval uses its left extension.
                bool left=v.lower_double()<1;
                require((left&&v.upper_double()<=1)||(!left&&v.lower_double()>=1),
                        "primal flow cell crosses reflection point");
                I t=left?v:I(2)-v;
                auto shape=primal_shape(par,t,n);
                QuadraticODE ode{shape.P*square(shape.n)*scale,shape.P*I(-2),
                                 linear?Jet(n): -shape.P};
                if(!left)for(unsigned a=1;a<=n;a+=2) {
                    ode.C[a]=-ode.C[a];ode.A[a]=-ode.A[a];ode.B[a]=-ode.B[a];
                }
                return ode;
            },order,flow_tolerance);
            std::string file=directory+"/filter-"+std::to_string(j)+".txt";
            if(replay)flow->load_and_verify(file,0,2,I(0));
            else {flow->segment(0,1,I(0));flow->segment(1,2,flow->endpoint);flow->save(file);}
            filters.push_back(std::move(flow));
        }
    }

    Jet left_filter_jet(unsigned index,const I &t,unsigned degree) const {
        require(contained(t,hull(I(0),I(1))),"left primal filter outside its half");
        I value=filters.at(index)->range(t);
        if(degree==0)return Jet(0,value);
        // At t=1 the global flow starts its reflected half. Integrals on
        // the left half require the left derivative extension there.
        auto shape=primal_shape(par,t,degree-1);
        bool linear=index==par.scales.size();
        I scale=linear?par.k:par.scales.at(index);
        return ode_jet({shape.P*square(shape.n)*scale,shape.P*I(-2),
                        linear?Jet(degree-1):-shape.P},value,degree);
    }

    std::vector<Jet> integrand(const I &t,unsigned degree) const {
        auto sh=primal_shape(par,t,degree);
        unsigned d=par.dimension();
        auto basis=chebyshev_basis(sh.t*I(2)-Jet(degree,I(1)),par.coefficients.size()-1);
        std::vector<Jet> values{sh.P*square(sh.n)*I(2),sh.P*sh.n*I(2),
                                sh.P*cube(sh.n)*I(2)};
        std::vector<std::vector<Jet>> jac(d,std::vector<Jet>(d,Jet(degree)));
        Jet z=sqrt((sh.a+sh.t*sh.h)/par.k);
        for(unsigned i=0;i<d;i++) {
            const Jet &phi=basis[par.directions[i]];
            jac[0][i]=sh.P*sh.Q*phi*(I(-2)/par.k);
            jac[1][i]=-sh.P*sh.t*(sh.a+sh.t*sh.h*I(2))*phi/(z*par.k);
        }
        for(unsigned j=0;j<filters.size();j++) {
            Jet r=left_filter_jet(j,t,degree);
            Jet R=filters[j]->jet(I(2)-t,degree);
            for(unsigned a=1;a<=degree;a+=2)R[a]=-R[a];
            bool linear=j==par.scales.size();
            values.push_back(sh.P*(linear?square(r):r)*I(2));
            Jet coefficient=linear?sh.P*(r*R*I(2)-sh.Q*(r+R))
                :sh.P*(sh.Q*(par.scales[j]/par.k)+r*R)*I(-2)/(Jet(degree,I(2))+r+R);
            for(unsigned i=0;i<d;i++)jac[2+j][i]=coefficient*basis[par.directions[i]];
        }
        for(const auto &row:jac)for(const auto &entry:row)values.push_back(entry);
        return values;
    }

    std::vector<I> cell_integral(double left,double right) const {
        I a(left),b(right),m=(a+b)/I(2),h=(b-a)/I(2);
        auto center=integrand(m,order-1),whole=integrand(hull(a,b),order);
        std::vector<I> out(center.size());
        for(unsigned i=0;i<out.size();i++) {
            for(unsigned j=0;j<order;j+=2)out[i]=out[i]+center[i][j]*I(2)*pow(h,j+1)/I(int(j+1));
            out[i]=out[i]+centered(abs_bound(whole[i][order])*I(2)*pow(h,order+1)/I(int(order+1)));
            numerical_trace()<<"component "<<i<<"\n";
            trace_cell("primal",left,right,center[i],whole[i],out[i]);
        }
        return out;
    }

    void integrate(const std::vector<std::pair<double,double>> &fixed={}) {
        numerical_trace()<<"primal-integral fixed "<<!fixed.empty()<<"\n";
        unsigned d=par.dimension();integrals.assign(1+d+d*d,I(0));partition.clear();
        std::vector<std::pair<double,double>> pending=fixed;
        if(pending.empty())pending.push_back({0,1});
        else std::reverse(pending.begin(),pending.end());
        while(!pending.empty()) {
            auto [left,right]=pending.back();pending.pop_back();
            bool accept=!fixed.empty();std::vector<I> value;
            try {
                value=cell_integral(left,right);
                if(fixed.empty()) {
                    accept=true;
                    for(const I &v:value)if(mpfr_cmp(width(v).hi.x,(quad_tolerance*(I(right)-I(left))).lo.x)>0) {
                        accept=false;break;
                    }
                }
            } catch(const std::ios_base::failure &) {throw;}
              catch(const std::runtime_error &) {if(!fixed.empty())throw;accept=false;}
            if(accept) {
                require(value.size()==integrals.size(),"primal quadrature dimension mismatch");
                for(unsigned j=0;j<value.size();j++)integrals[j]=integrals[j]+value[j];
                numerical_trace()<<"accept "<<left<<" "<<right<<"\n";
                partition.push_back({left,right});
            } else {
                numerical_trace()<<"subdivide "<<left<<" "<<right<<"\n";
                double mid=left+(right-left)/2;
                require(mid>left&&mid<right,"primal quadrature exhausted time coordinates");
                pending.push_back({mid,right});pending.push_back({left,mid});
            }
        }
        numerical_trace()<<"integrals";
        for(const auto &v:integrals)numerical_trace()<<" "<<v;
        numerical_trace()<<"\n"<<std::flush;
        for(unsigned j=0;j<filters.size();j++) {
            I r=filters[j]->range(I(1));
            integrals[3+j]=integrals[3+j]+(j==par.scales.size()?square(r):log(I(1)+r));
        }
    }
    std::vector<I> residual() const {
        std::vector<I> result{integrals[0]-I(1),integrals[1]-I(1)};
        for(unsigned j=0;j<par.scales.size();j++) {
            I x=sqrt(par.scales[j]);
            result.push_back(integrals[3+j]-x-log(I(1)+exp(I(-2)*x))+log(I(2)));
        }
        if(par.second)result.push_back(integrals[3+par.scales.size()]-square(par.k)/I(6));
        return result;
    }
    std::vector<I> correction() const {
        auto f=residual();std::vector<I> out(f.size());
        for(unsigned i=0;i<f.size();i++)for(unsigned j=0;j<f.size();j++)out[i]=out[i]+par.C[i][j]*f[j];
        return out;
    }
    I contraction() const {
        unsigned d=par.dimension();I bound;
        for(unsigned i=0;i<d;i++) {
            I row;
            for(unsigned j=0;j<d;j++) {
                I entry(int(i==j));
                for(unsigned a=0;a<d;a++)entry=entry-par.C[i][a]*integrals[1+d+a*d+j];
                row=row+abs_bound(entry);
            }
            bound=upper(hull(bound,row));
        }
        return bound;
    }
};

inline I infinity_bound(const std::vector<I> &v) {
    I value;for(const I &x:v)value=upper(hull(value,abs_bound(x)));return value;
}

inline void save_primal_partition(const std::string &file,
                 const std::vector<std::pair<double,double>> &partition) {
    auto out=cert::output_file(file);out<<"PRIMAL_QUADRATURE 1\n"<<partition.size()<<"\n"<<std::hexfloat;
    for(auto [left,right]:partition)out<<left<<" "<<right<<"\n";
    finish_output(out);
}

inline std::vector<std::pair<double,double>> load_primal_partition(const std::string &file) {
    std::ifstream in(file);std::string magic,l,r,extra;in>>magic;
    unsigned version=read_count(in),count=read_count(in);
    require(magic=="PRIMAL_QUADRATURE"&&version==1&&count>0,"invalid primal quadrature partition");
    std::vector<std::pair<double,double>> partition;double previous=0;
    for(std::size_t j=0;j<count;j++) {
        in>>l>>r;std::size_t used_l,used_r;double left=std::stod(l,&used_l),right=std::stod(r,&used_r);
        require(used_l==l.size()&&used_r==r.size()&&std::isfinite(left)&&std::isfinite(right),
                "invalid primal partition coordinate");
        require(contained(I(l),I(left))&&contained(I(r),I(right)),"inexact primal partition coordinate");
        require(left==previous&&right>left&&right<=1,"primal quadrature partition has gap or overlap");
        partition.push_back({left,right});previous=right;
    }
    require(bool(in)&&previous==1&&!(in>>extra),"incomplete or trailing primal partition");
    return partition;
}

inline void report_primal(const std::string &file,const PrimalEvaluation &center,
                          const PrimalEvaluation &cube,const I &rho) {
    I residual=infinity_bound(center.correction()),r=cube.contraction();
    // |d log(P n^3)/dp|=|(D-2Q)/(2(D+Q))|<=1, and every
    // selected Chebyshev polynomial has modulus <=1 on [-1,1].
    I ceiling=center.integrals[2]*exp(rho*I(int(center.par.dimension())));
    auto out=cert::output_file(file);out<<"PRIMAL_VALUES 1\n";
    out<<"radius "<<rho<<"\nresidual "<<residual<<"\ncontraction "<<r
       <<"\ncentral_energy "<<center.integrals[2]<<"\nenergy_ceiling "<<ceiling<<"\n";
    auto f=center.residual();
    for(unsigned j=0;j<f.size();j++)out<<"F "<<j<<" "<<f[j]<<"\n";
    unsigned d=center.par.dimension();
    for(unsigned i=0;i<d;i++)for(unsigned j=0;j<d;j++)
        out<<"J "<<i<<" "<<j<<" "<<cube.integrals[1+d+i*d+j]<<"\n";
    finish_output(out);
    require(rho.positive()&&width(rho).zero(),"cube radius is not an exact positive point");
    require((I(1)-r).positive(),"primal Jacobian does not contract on the whole cube");
    require((rho-residual-r*rho).positive(),"primal cube does not contain its image strictly");
    check_runtime();
    std::cout<<"accepted exact primal cube: radius "<<rho<<" contraction "<<r
             <<" energy ceiling "<<ceiling<<"\n";
}
} // namespace cert
