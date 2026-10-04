// Produce the affine price and complete state cover of lem:regular-criterion.
#include "state_cover.hpp"
#include "compact_cover.hpp"
#include <filesystem>

using namespace transport;

struct CellResult {
    Price price;
    Cover::Maximum cover;
    double midpoint_price=0;
    std::size_t evaluations=0;
};

struct ScaleRepair : std::runtime_error {
    double ratio;
    explicit ScaleRepair(double r):std::runtime_error("homothetic price repair"),ratio(r){}
};

struct Progress {
    double last_width=0,last_dilation=1;
    Price last_price;
};

static double repair_ratio(const StateFormula &state,double tolerance) {
    double b=(state.b.v.lo+state.b.v.hi)/2;
    double c=(state.c.v.lo+state.c.v.hi)/2;
    double ell=(state.ell.v.lo+state.ell.v.hi)/2;
    // Minimum over all prices for this one state is
    // -c-(1-ell)b-1/2+3ell/2-ell^(3/2).
    // This calculation proposes a scale only. Every new state box is
    // subsequently proved, and earlier boxes use convex homothety.
    double cost=-c-(1-ell)*b;
    double allowance=.5-1.5*ell+ell*std::sqrt(std::max(0.,ell));
    double reserve=tolerance*(1+std::abs(b)+std::abs(c));
    double ratio=cost>0?(allowance-reserve)/cost:.5;
    if(!(ratio>0))ratio=.5;
    ratio=std::min(ratio,1-std::sqrt(DBL_EPSILON));
    return ratio;
}

static CellResult produce(const Parameters &par,const Profile &whole,const Profile &middle,
                          double price_tolerance,std::size_t budget,
                          const Price *warm=nullptr,const std::vector<Rectangle> &seed={}) {
    Cover at_point(par,middle,middle);
    Price e; e.center=middle.t.v.lo;
    if(warm) e.intercept=warm->intercept+warm->slope*(e.center-warm->center);
    StateKernel point_kernel(par,middle);
    CellResult out;
    std::size_t midpoint_leaves=2;
    std::size_t point_budget=budget;
    std::vector<Rectangle> point_partition=seed;
    for(unsigned iteration=0;iteration<DBL_MANT_DIG;iteration++) {
        Cover::Maximum maximum;
        for(;;) {
            maximum=at_point.maximum(e,price_tolerance/8,point_budget,false,point_partition);
            if(!maximum.budget_exhausted) break;
            // Expand a work budget only after exhausting its complete
            // state partition. Reuse the geometry; every bound is recomputed.
            cert::require(point_budget<=std::numeric_limits<std::size_t>::max()/2,
                          "point state work capacity exhausted");
            point_budget*=2;point_partition.clear();
            for(const auto &leaf:maximum.leaves)point_partition.push_back(leaf.box);
            std::cout<<"increase point state budget "<<point_budget<<" at t "
                     <<std::setprecision(17)<<middle.t.v.lo<<"\n"<<std::flush;
        }
        point_partition.clear();
        for(const auto &leaf:maximum.leaves)point_partition.push_back(leaf.box);
        midpoint_leaves=std::max(midpoint_leaves,maximum.leaves.size());
        if(maximum.exhausted) throw std::runtime_error("midpoint state budget exhausted");
        auto pt=maximum.active.point;
        auto state=point_kernel.evaluate(D(pt[0]),D(pt[1]),maximum.active.box.branch);
        double denominator=price_derivative(state,e.intercept);
        if(!(denominator>0)) {
            if(maximum.upper<0) {
                // A strictly feasible price is already available. At an
                // active branch with the opposite derivative, increasing
                // it is not a valid repair proposal; retain it unchanged.
                out.midpoint_price=e.intercept;
                break;
            }
            std::cerr<<std::setprecision(17)<<"price diagnostic t "<<middle.t.v.lo
                     <<" e "<<e.intercept<<" upper "<<maximum.upper
                     <<" b "<<state.b.v.lo<<" "<<state.b.v.hi
                     <<" c "<<state.c.v.lo<<" "<<state.c.v.hi
                     <<" ell "<<state.ell.v.lo<<" "<<state.ell.v.hi<<"\n";
            throw ScaleRepair(repair_ratio(state,price_tolerance));
        }
        double increment=maximum.upper/denominator;
        if(std::abs(maximum.upper)<=price_tolerance*denominator/2) {
            // d/dt F(t,e(t))=F_t-F_e^abs e', with F_e^abs>0
            // only at this proposal point. No global monotonicity is used.
            A residual=state.residual(e.at(middle.t.v));
            e.slope=(residual.g[2].lo+residual.g[2].hi)/(2*denominator);
            // A warm proposal must also be allowed to decrease. Otherwise
            // adding the safety reserve on every adjacent cell accumulates
            // an unnecessary price. Only the final whole-cell cover decides
            // acceptance, so both signs of this Newton proposal are safe.
            e.intercept+=increment+price_tolerance;
            out.midpoint_price=e.intercept;
            break;
        }
        e.intercept+=increment;
        cert::require(iteration+1<DBL_MANT_DIG,"price proposal failed to converge");
    }
    Cover cover(par,whole,middle);
    // A failed spatial cell is refined; spending orders of magnitude
    // more state work than at its midpoint cannot cure spatial width.
    out.cover=cover.maximum(e,price_tolerance,std::min(point_budget,2*midpoint_leaves),true,point_partition);
    out.price=e;
    out.evaluations+=at_point.evaluations+cover.evaluations;
    return out;
}

int main(int argc,char **argv) {
    try {
        cert::require(argc==12||argc==14||argc==15,
          "usage: produce_transport reference polynomial cache bits order flow_tol price_tol quadrature_tol state_budget output [half left [right]] produce|probe|segment");
        cert::precision=cert::count_argument(argv[4]); cert::require(cert::precision>=64,"precision too low");
        unsigned order=cert::count_argument(argv[5]); cert::I flow_tol(argv[6]),quad_tol(argv[8]);
        double tolerance=std::stod(argv[7]); std::size_t budget=cert::count_argument(argv[9]);
        cert::require(std::isfinite(tolerance)&&tolerance>0,"finite positive price accuracy required");
        cert::check_runtime(); boxes::check_runtime();
        std::filesystem::create_directories(argv[10]);
        std::string cache=argv[3],prefix=std::string(argv[10])+"/";
        if(cache.back()!='/')cache+='/';
        // Proposal construction uses the prerequisite flow enclosures.
        // Final acceptance repeats their differential predicates in
        // verify_transport, independently of this proposal calculation.
        cert::Reference ref(cert::Parameters(argv[1]),order,flow_tol,cache+"support-filter-flow.txt",true);
        ref.build_filters("",cache);
        if(std::filesystem::exists(cache+"complement-0.txt"))ref.build_complements("",cache);
        Parameters par(argv[2]);
        cert::require(ref.par.scales.size()==2,"two profile scales required");
        for(unsigned j=0;j<2;j++)
            cert::require(transport::convert(ref.par.scales[j]).lo==par.scales[j].lo
                          &&transport::convert(ref.par.scales[j]).hi==par.scales[j].hi,"filter scale mismatch");
        bool probe=std::string(argv[argc-1])=="probe";
        bool segment=std::string(argv[argc-1])=="segment";
        cert::require(probe||segment||std::string(argv[argc-1])=="produce","unknown producer mode");
        if(probe) {
            cert::require(argc==14,"probe requires half and t coordinate");
            unsigned half=cert::count_argument(argv[11]); double t=std::stod(argv[12]);
            cert::require(half<2&&std::isfinite(t)&&t>=0&&t<=1,"invalid probe coordinate");
            Profile middle(ref,I(t),half);
            StateKernel optimized(par,middle);
            double discrepancy=0;
            for(unsigned branch=0;branch<2;branch++) for(double u:{0.,.25,.5,.75,1.})
                for(double v:{0.,.25,.5,.75,1.}) {
                    auto a=optimized.evaluate(D(u),D(v),branch);
                    auto b=StateFormula::build(par,middle,D(u),D(v),branch);
                    for(auto pair:{std::pair{a.b.v,b.b.v},std::pair{a.c.v,b.c.v}}) {
                        cert::require(pair.first.lo<=pair.second.hi&&pair.second.lo<=pair.first.hi,
                                      "optimized kernel disagrees with uncached formula");
                        discrepancy=std::max(discrepancy,boxes::magnitude(pair.first-pair.second));
                    }
                }
            std::cout<<std::setprecision(17)<<"kernel agreement "<<discrepancy<<"\n"<<std::flush;
            CellResult result=produce(par,middle,middle,tolerance,budget);
            std::cout<<"probe "<<half<<" "<<t<<" price "<<result.price.intercept
                     <<" slope "<<result.price.slope<<" upper "<<result.cover.upper
                     <<" lower "<<result.cover.lower<<" leaves "<<result.cover.leaves.size()
                     <<" evaluations "<<result.evaluations<<" accepted "<<result.cover.accepted<<"\n";
            cert::finish_trace(); return result.cover.accepted?0:2;
        }
        cert::require(argc==(segment?15:12),"unexpected producer arguments");
        unsigned first_half=0,last_half=2;
        double begin=0,end=1;
        if(segment) {
            first_half=cert::count_argument(argv[11]);last_half=first_half+1;
            auto dyadic=[](const char *arg) {
                std::string s(arg);std::size_t used=0;double x=std::stod(s,&used);
                cert::require(used==s.size()&&std::isfinite(x)&&s.rfind("0x",0)==0
                              &&s.find('p')!=std::string::npos,"segment requires exact hexadecimal coordinates");
                I exact(s);
                cert::require(cert::contained(exact,I(x))&&cert::contained(I(x),exact),
                              "segment endpoint is not exactly binary64 representable");
                return x;
            };
            begin=dyadic(argv[12]);end=dyadic(argv[13]);
            cert::require(first_half<2&&0<=begin&&begin<end&&end<=1,"invalid spatial segment");
        }
        Progress progress;
        par.dilation=D(1);
        PackedCoverWriter witness(prefix+"state-cover.bin",segment,first_half,begin,end);
        auto values=cert::output_file(prefix+"transport-values.txt");
        values<<(segment?"TRANSPORT_VALUES_PART 1\n":"TRANSPORT_VALUES 2\n");
        I total;
        std::size_t count=0,leaves=0,evaluations=0;
        for(unsigned half=first_half;half<last_half;half++) {
            std::vector<std::pair<double,double>> pending{{begin,end}};
            while(!pending.empty()) {
                auto [left,right]=pending.back(); pending.pop_back();
                double mid=cell_center(left,right);
                bool accepted=false;
                try {
                    Profile whole(ref,cert::hull(I(left),I(right)),half),middle(ref,I(mid),half);
                    D weight=boxes::pow(whole.t.v,3)*whole.z.v*D(3)/whole.k;
                    cert::require(weight.hi>0,"zero spatial price weight");
                    // The global accuracy is expressed in the integral,
                    // whose Jacobian vanishes cubically at the endpoint.
                    // This work allocation is computed from that Jacobian;
                    // the final integral is still bounded independently.
                    double local_tolerance=std::min(std::sqrt(tolerance),tolerance/weight.hi);
                    bool nearby=progress.last_width>0&&std::abs(mid-progress.last_price.center)<=2*progress.last_width;
                    Price warm=progress.last_price;
                    double factor=par.dilation.lo/progress.last_dilation;
                    warm.intercept*=factor;warm.slope*=factor;
                    CellResult result=produce(par,whole,middle,local_tolerance,budget,
                        nearby?&warm:nullptr);
                    evaluations+=result.evaluations;
                    if(result.cover.accepted) {
                        auto integral=cert::quadrature([&](const I &t,unsigned n) {
                            auto sh=ref.local_shape(t,n,true);
                            Jet e=Jet(n,I(result.price.intercept))
                                 +(cert::variable(t,n)-Jet(n,I(mid)))*I(result.price.slope);
                            return cert::cube(sh.t)*sh.z*e*(I(3)/ref.par.k);
                        },left,right,order,quad_tol);
                        total=total+integral.value/I(par.dilation.lo);
                        witness.cell(half,left,right,par.dilation.lo,result.price.intercept,result.price.slope,
                                     negative_margin(result.cover.upper),result.cover.leaves,integral.value);
                        leaves+=result.cover.leaves.size();++count;accepted=true;
                        progress.last_price=result.price;progress.last_width=right-left;
                        progress.last_dilation=par.dilation.lo;
                        std::cout<<std::setprecision(12)<<"cell "<<half<<" "<<left<<" "<<right
                                 <<" price "<<result.price.intercept<<" leaves "<<result.cover.leaves.size()
                                 <<" dilation "<<par.dilation.lo
                                 <<" total "<<(total*I(par.dilation.lo)).lower_double()
                                 <<" "<<(total*I(par.dilation.lo)).upper_double()
                                 <<" evaluations "<<evaluations<<"\n"<<std::flush;
                    }
                } catch(const ScaleRepair &repair) {
                    double before=par.dilation.lo;
                    double after=std::nextafter(before*repair.ratio,0.);
                    cert::require(after>0&&after<before,"homothetic scale exhausted binary64");
                    par.dilation=D(after);
                    std::cout<<std::setprecision(17)<<"repair dilation "<<before<<" -> "<<after
                             <<" at half "<<half<<" cell "<<left<<" "<<right<<"\n"<<std::flush;
                    pending.emplace_back(left,right);
                    continue;
                } catch(const std::ios_base::failure &) { throw;
                } catch(const std::runtime_error &e) {
                    std::string why=e.what();
                    if(why.rfind("midpoint ",0)==0||why.rfind("price proposal ",0)==0)
                        throw;
                    std::cout<<"refine "<<half<<" "<<std::setprecision(17)<<left<<" "<<right
                             <<" reason "<<e.what()<<"\n"<<std::flush;
                }
                if(!accepted) {pending.emplace_back(mid,right);pending.emplace_back(left,mid);}
            }
            progress.last_width=0;
        }
        witness.finish(count,leaves,par.dilation.lo);
        values<<"dilation "<<I(par.dilation.lo)<<"\nprice "<<total*I(par.dilation.lo)
              <<"\nundilated_price "<<total<<"\n"<<"cells "<<count<<"\n"<<"leaves "<<leaves<<"\n";
        cert::finish_output(values);
        cert::check_runtime();boxes::check_runtime();
        std::cout<<"complete transport proposal; independent flow and cover replay required\n";
        cert::finish_trace(); return 0;
    } catch(const std::exception &e) {
        std::cerr<<"REJECT: "<<e.what()<<"\n";return 1;
    }
}
