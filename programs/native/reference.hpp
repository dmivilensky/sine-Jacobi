#pragma once
// Positive rational support and its contact reference (thm:contact).
// The substitution q=t^3 gives a regular positive cubic in z=n/t.
// Reflection uses the appropriate one-sided derivatives at t=1. Complementary
// flows and cubic-zero removal implement lem:regular-criterion and
// lem:cubic-quotient; their identities follow from uniqueness of the ODE.
#include "flow.hpp"
#include "quadrature.hpp"
#include <memory>
#include <filesystem>

namespace cert {
struct Parameters {
    I k,beta,b,gamma;
    std::vector<I> support,scales;
    explicit Parameters(const std::string &path) {
        k=square(pi())/I(4);
        std::ifstream in(path); require(bool(in),"cannot open parameters");
        std::string magic,s; in>>magic;
        unsigned version=read_count(in),count;
        require(magic=="CONTACT_REFERENCE"&&version==1,"invalid parameter format");
        in>>s; beta=I(s); in>>s; b=I(s); in>>s; gamma=I(s);
        count=read_count(in); require(count>0&&(count&1),"invalid rational-profile dimension");
        for(unsigned i=0;i<count;i++) { in>>s; support.emplace_back(s); }
        count=read_count(in);
        for(unsigned i=0;i<count;i++) { in>>s; scales.emplace_back(s); }
        require(bool(in),"truncated parameters");
        require_end(in);
        require(beta.nonnegative()&&b.positive()&&gamma.nonnegative(),"inadmissible base parameters");
        for(const I &x:support) require(x.positive(),"rational profile coefficient must be positive");
        for(const I &x:scales) require(x.positive(),"filter scale must be positive");
        for(unsigned j=1;j<scales.size();j++)require((scales[j]-scales[j-1]).positive(),"unordered filter scales");
    }
};
struct Shape { Jet t,q,Q,z,n,P; };

inline Jet reflected(Jet a) {
    for(unsigned j=1;j<=a.order();j+=2) a[j]=-a[j];
    return a;
}

inline Jet ground_coordinate(const I &v,unsigned n) {
    bool right=v.lower_double()>=1;
    require(right||v.upper_double()<=1,"ground-coordinate cell crosses symmetry point");
    Jet t=right?Jet(n,I(2))-variable(v,n):variable(v,n);
    return right?Jet(n,I(2))-cube(t):cube(t);
}

class Reference {
public:
    Parameters par;
    unsigned order;
    I flow_tol;
    bool proposal_cache;
    std::unique_ptr<Flow> supporting_filter;
    std::unique_ptr<Flow> support_complement;
    mutable std::unique_ptr<CubicQuotient> residual_quotient;
    std::vector<std::unique_ptr<Flow>> filters;
    std::unique_ptr<Flow> linear_filter;
    mutable std::vector<std::unique_ptr<CubicQuotient>> filter_quotients;
    mutable std::unique_ptr<CubicQuotient> linear_quotient;
    std::vector<std::unique_ptr<Flow>> complements;
    mutable std::vector<std::unique_ptr<CubicQuotient>> complement_quotients;

    Reference(Parameters p,unsigned n,I tol,const std::string &verified_cache="",bool proposal_only=false)
        : par(std::move(p)),order(n),flow_tol(tol),proposal_cache(proposal_only) {
        supporting_filter=std::make_unique<Flow>([this](const I &v,unsigned n) {
            Shape sh=support_shape(v,n);
            return QuadraticODE{sh.P*square(sh.n)*par.k,
                                sh.P*I(-2),-sh.P*par.beta};
        },order,flow_tol);
        if(verified_cache.empty()) {
            supporting_filter->segment(0,1,I(0));
            supporting_filter->segment(1,2,supporting_filter->endpoint);
        } else read_cache(*supporting_filter,verified_cache,0,2,I(0));
        if(!verified_cache.empty()) {
            std::string folder=std::filesystem::path(verified_cache).parent_path().string()+"/";
            if(std::filesystem::exists(folder+"support-complement-end.txt"))build_support_complement("",folder);
        }
    }

    void read_cache(Flow &flow,const std::string &path,double begin,double end,I initial) {
        if(proposal_cache)flow.load_proposal(path,begin,end,initial);
        else flow.load_and_verify(path,begin,end,initial);
    }

    void build_support_complement(const std::string &output_prefix="",const std::string &cache_prefix="") {
        require(!support_complement,"support complement already constructed");
        support_complement=std::make_unique<Flow>([this](const I &t,unsigned n) {
            Shape sh=support_shape(t,n,true);Jet q=cube(variable(t,n));
            return QuadraticODE{sh.P*square(q)*(I(1)+par.beta),
                                sh.P*(Jet(n,I(1))+q*par.beta)*I(-2),sh.P*par.beta};
        },order,flow_tol);
        double end=1;
        if(!cache_prefix.empty()&&std::filesystem::exists(cache_prefix+"support-complement-end.txt")) {
            std::ifstream in(cache_prefix+"support-complement-end.txt");std::string token;in>>token;
            require(bool(in),"missing complementary support endpoint");
            std::size_t used=0;end=std::stod(token,&used);
            require(used==token.size()&&end>0&&end<=1&&contained(I(token),I(end)),
                    "invalid complementary support endpoint");
            require_end(in);
            read_cache(*support_complement,cache_prefix+"support-complement.txt",0,end,I(0));
        } else {
            // Switch descriptions after cancellation is no longer severe.
            // The switch point is an accepted flow endpoint, computed by
            // the comparison 2R>=q, not a prescribed spatial cutoff.
            end=support_complement->segment(0,1,I(0),1.,[](double t,const I &value) {
                return (I(2)*value-pow(I(t),3)).nonnegative();
            });
        }
        if(!output_prefix.empty()) {
            support_complement->save(output_prefix+"support-complement.txt");
            auto out=output_file(output_prefix+"support-complement-end.txt");out<<std::hexfloat<<end<<"\n";
            finish_output(out);
        }
        residual_quotient.reset();
        std::cerr<<"complementary support coordinate reaches "<<end<<"\n";
    }

    Shape support_shape(const I &v,unsigned n,bool force_left=false) const {
        require(contained(v,hull(I(0),I(2))),"support coordinate outside [0,2]");
        bool right=!force_left&&v.lower_double()>=1;
        require(right||v.upper_double()<=1,"Taylor cell crosses the symmetry point");
        Jet t=right?Jet(n,I(2))-variable(v,n):variable(v,n);
        Jet small=cube(t),h=Jet(n,I(2))-small,Q=small*h;
        Jet p(n,par.support[0]);
        for(unsigned j=1;j<par.support.size();j+=2)
            p=p+Q/(Q+Jet(n,par.support[j+1]))*par.support[j];
        Jet z=cubic(t*h*(I(3)/par.k),p*h*(I(2)/par.k));
        Jet P=inverse(t*h*I(2)+p*h/z*I(2))*I(3);
        return Shape{t,right?h:small,Q,z,t*z,P};
    }

    Jet supporting_left_jet(const I &t,unsigned n) const {
        require(contained(t,hull(I(0),I(1))),"left supporting jet outside its closed half");
        I value=supporting_filter->range(t);
        if(n==0)return Jet(0,value);
        // The global flow chooses right derivatives at its shared initial
        // node v=1. Gamma is parametrized by the LEFT variable t up to and
        // including t=1, so its derivatives must instead use this left ODE.
        Shape sh=support_shape(t,n-1,true);
        return ode_jet(QuadraticODE{sh.P*square(sh.n)*par.k,
                                   sh.P*I(-2),-sh.P*par.beta},value,n);
    }

    Jet left_residual_jet(const I &t,unsigned n) const {
        if(support_complement&&t.upper_double()<=support_complement->cells.back().right)
            return nonnegative_value(support_complement->jet(t,n));
        return nonnegative_value(cube(variable(t,n))-supporting_left_jet(t,n));
    }

    Jet residual_over_cube(const I &t,unsigned n) const {
        double cutoff=supporting_filter->cells.front().right;
        if(t.upper_double()<=cutoff) {
            if(!residual_quotient||residual_quotient->order()<n)
                residual_quotient=std::make_unique<CubicQuotient>(
                    [this](const I &v,unsigned d){return left_residual_jet(v,d);},
                    2*std::max(order,n)+4,cutoff);
            return residual_quotient->evaluate(t,n);
        }
        if(t.positive()) return left_residual_jet(t,n)/cube(variable(t,n));
        require(t.lower_double()==0&&t.upper_double()<=1,"invalid endpoint quotient");
        // R(0)=R'(0)=R''(0)=0. For each j, the j-th normalized
        // derivative of R(t)/t^3 is a weighted average of R^(j+3)/(j+3)!
        // over [0,t], with nonnegative weights of total one. This includes
        // t=0 and avoids any division by an interval containing zero.
        Jet high=left_residual_jet(t,n+3),out(n);
        for(unsigned j=0;j<=n;j++) out[j]=high[j+3];
        return out;
    }

    Jet gamma_ratio(const I &t,unsigned n) const {
        require(contained(t,hull(I(0),I(1))),"contact coordinate outside left half");
        Jet u=variable(t,n),q=cube(u),h=Jet(n,I(2))-q,Q=q*h;
        Jet wl=nonnegative_value(supporting_left_jet(t,n));
        Jet wr=nonnegative_value(reflected(supporting_filter->jet(I(2)-t,n)));
        Jet ratio=residual_over_cube(t,n);
        return nonnegative_value((q*(h-wr)+ratio*(wr*I(2)-Q))/
               (h*(Jet(n,I(2))+(wl+wr)*par.beta)));
    }

    Shape local_shape(const I &v,unsigned n,bool force_left=false) const {
        bool right=!force_left&&v.lower_double()>=1;
        require(right||v.upper_double()<=1,"local Taylor cell crosses symmetry point");
        I u=right?I(2)-v:v;
        Jet t=variable(u,n),q=cube(t),h=Jet(n,I(2))-q,Q=q*h;
        Jet p=Jet(n,par.b)+gamma_ratio(u,n)*(par.k*par.gamma);
        Jet z=cubic(t*h*(I(3)/par.k),p*h*(I(2)/par.k));
        Jet P=inverse(t*h*I(2)+p*h/z*I(2))*I(3),amp=t*z;
        if(right) return Shape{reflected(t),reflected(h),reflected(Q),
                                reflected(z),reflected(amp),reflected(P)};
        return Shape{t,q,Q,z,amp,P};
    }

    void build_filters(const std::string &output_prefix="",const std::string &cache_prefix="") {
        for(unsigned index=0;index<par.scales.size();index++) {
            I s=par.scales[index];
            auto f=std::make_unique<Flow>([this,s](const I &v,unsigned n) {
                Shape sh=local_shape(v,n);
                return QuadraticODE{sh.P*square(sh.n)*s,sh.P*I(-2),-sh.P};
            },order,flow_tol);
            std::string name="filter-"+std::to_string(index)+".txt";
            if(!cache_prefix.empty()&&std::filesystem::exists(cache_prefix+name))
                read_cache(*f,cache_prefix+name,0,2,I(0));
            else { f->segment(0,1,I(0)); f->segment(1,2,f->endpoint); }
            if(!output_prefix.empty()) f->save(output_prefix+name);
            filters.push_back(std::move(f));
            filter_quotients.emplace_back();
        }
        linear_filter=std::make_unique<Flow>([this](const I &v,unsigned n) {
            Shape sh=local_shape(v,n);
            return QuadraticODE{sh.P*square(sh.n)*par.k,sh.P*I(-2),Jet(n)};
        },order,flow_tol);
        if(!cache_prefix.empty()&&std::filesystem::exists(cache_prefix+"linear-filter.txt"))
            read_cache(*linear_filter,cache_prefix+"linear-filter.txt",0,2,I(0));
        else {
            linear_filter->segment(0,1,I(0));
            linear_filter->segment(1,2,linear_filter->endpoint);
        }
        if(!output_prefix.empty()) linear_filter->save(output_prefix+"linear-filter.txt");
    }

    void build_complements(const std::string &output_prefix="",const std::string &cache_prefix="") {
        require(complements.empty(),"complementary flows already constructed");
        // For delta_s=q-k*r_s/s, cancellation of the common forcing gives
        // delta_s'=(1+s/k)q^2-2(1+s*q/k)delta_s+(s/k)delta_s^2.
        // For the linear filter the corresponding equation is
        // delta_0'=q^2-2*delta_0. These small residuals are O(t^7), so
        // evaluating 1-delta_s/t^3 preserves the leading constant exactly.
        // Equality with the original filter follows from ODE uniqueness;
        // it is not inferred from agreement of numerical integrations.
        for(unsigned index=0;index<=par.scales.size();index++) {
            I a=index==par.scales.size()?I(0):par.scales[index]/par.k;
            auto f=std::make_unique<Flow>([this,a](const I &t,unsigned n) {
                Shape sh=local_shape(t,n,true);Jet q=cube(variable(t,n));
                return QuadraticODE{sh.P*square(q)*(I(1)+a),
                                    sh.P*(Jet(n,I(1))+q*a)*I(-2),sh.P*a};
            },order,flow_tol);
            std::string name="complement-"+std::to_string(index)+".txt";
            if(!cache_prefix.empty()&&std::filesystem::exists(cache_prefix+name))
                read_cache(*f,cache_prefix+name,0,1,I(0));
            else f->segment(0,1,I(0));
            if(!output_prefix.empty())f->save(output_prefix+name);
            complements.push_back(std::move(f));complement_quotients.emplace_back();
        }
    }

    Jet local_filter_left_jet(const I &t,unsigned n,int index) const {
        const Flow &f=index<0?*linear_filter:*filters.at(unsigned(index));
        I value=f.range(t);if(n==0)return Jet(0,value);
        Shape sh=local_shape(t,n-1,true);
        I scale=index<0?par.k:par.scales.at(unsigned(index));
        return ode_jet(QuadraticODE{sh.P*square(sh.n)*scale,sh.P*I(-2),
                                   index<0?Jet(n-1):-sh.P},value,n);
    }

    Jet normalized_filter(const I &t,unsigned n,int index) const {
        if(!complements.empty()) {
            unsigned j=index<0?par.scales.size():unsigned(index);
            const Flow &f=*complements.at(j);auto &model=complement_quotients.at(j);
            double cutoff=f.cells.front().right;Jet quotient(n);
            if(t.upper_double()<=cutoff) {
                if(!model||model->order()<n)
                    model=std::make_unique<CubicQuotient>(
                        [&f](const I &v,unsigned d){return f.jet(v,d);},order+n+4,cutoff);
                quotient=model->evaluate(t,n);
            } else if(t.positive())quotient=f.jet(t,n)/cube(variable(t,n));
            else {
                Jet high=f.jet(t,n+3);
                for(unsigned l=0;l<=n;l++)quotient[l]=high[l+3];
            }
            return (Jet(n,I(1))-quotient)*(index<0?I(1):par.scales[unsigned(index)]/par.k);
        }
        const Flow &f=index<0?*linear_filter:*filters.at(unsigned(index));
        auto &model=index<0?linear_quotient:filter_quotients.at(unsigned(index));
        double cutoff=f.cells.front().right;
        if(t.upper_double()<=cutoff) {
            if(!model||model->order()<n)
                model=std::make_unique<CubicQuotient>(
                    [this,index](const I &v,unsigned d){return local_filter_left_jet(v,d,index);},order+n+4,cutoff);
            return model->evaluate(t,n);
        }
        if(t.positive()) return local_filter_left_jet(t,n,index)/cube(variable(t,n));
        Jet high=local_filter_left_jet(t,n+3,index),out(n);
        for(unsigned j=0;j<=n;j++) out[j]=high[j+3];
        return out;
    }

    I target_H(const I &s) const {
        I x=sqrt(s);
        return x+log(I(1)+exp(-I(2)*x))-log(I(2));
    }
    I target_S() const {
        if(par.beta.zero())return square(par.k)/I(6);
        return par.k/par.beta-I(2)*target_H(par.k*par.beta)/square(par.beta);
    }
    I vacuum_square(const I &w) const {
        if(par.beta.zero())return square(w)/I(4);
        return (par.beta*w-I(2)*log(I(1)+par.beta*w/I(2)))/square(par.beta);
    }
};
} // namespace cert
