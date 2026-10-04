// Enclose ca:transport: contact value, masses, filters and exact reflection tails.
#include "reference.hpp"
#include <filesystem>

using namespace cert;
int main(int argc,char **argv) {
    try {
        require(argc==7||argc==8,"usage: verify_reference parameters bits order flow_tol quadrature_tol output_directory [support_cache]");
        precision=cert::count_argument(argv[2]); require(precision>=64,"at least 64 bits required");
        unsigned order=cert::count_argument(argv[3]);
        I flow_tol(argv[4]),quad_tol(argv[5]);
        check_runtime();
        std::filesystem::create_directories(argv[6]);
        std::string prefix=std::string(argv[6])+"/";
        Reference ref(Parameters(argv[1]),order,flow_tol,argc==8?argv[7]:"");
        if(!ref.support_complement) ref.build_support_complement(prefix);
        else {
            ref.support_complement->save(prefix+"support-complement.txt");
            auto endpoint=cert::output_file(prefix+"support-complement-end.txt");
            endpoint<<std::hexfloat<<ref.support_complement->cells.back().right<<"\n";
            finish_output(endpoint);
        }
        ref.supporting_filter->save(prefix+"support-filter-flow.txt");
        std::cout<<"support cells "<<ref.supporting_filter->cells.size()<<" rejected "
                 <<ref.supporting_filter->rejected<<"\n"<<std::flush;
        auto integrate=[&](Integrand f) {
            return quadrature(f,0,1,order,quad_tol);
        };
        Integral Jcore=integrate([&](const I &t,unsigned n) {
            Shape sh=ref.support_shape(t,n,true);
            Jet left=ref.supporting_left_jet(t,n);
            Jet right=reflected(ref.supporting_filter->jet(I(2)-t,n));
            return (ref.gamma_ratio(t,n)*sh.Q*I(2)-square(left)-square(right))*sh.P;
        });
        I boundary_filter=ref.supporting_filter->endpoint;
        I J=Jcore.value-ref.vacuum_square(boundary_filter);
        Integral cubic_mass=integrate([&](const I &t,unsigned n) {
            Shape sh=ref.local_shape(t,n,true); return square(sh.t)*sh.n;
        });
        I base=-ref.par.b+I(2)*ref.par.b/ref.par.k+I(9)*cubic_mass.value/ref.par.k
               -ref.par.gamma*(J+ref.target_S());
        auto report=cert::output_file(prefix+"reference-values.txt");
        report<<"CONTACT_VALUES 1\n";
        report<<"k "<<ref.par.k<<"\n"<<"J "<<J<<"\n";
        report<<"base "<<base<<"\n";
        report.flush();
        std::cout<<"base "<<base<<"\n"<<std::flush;
        for(unsigned power=1;power<=3;power++) {
            Integral mass=integrate([&](const I &t,unsigned n) {
                Shape sh=ref.local_shape(t,n,true); Jet amp(n,I(1));
                for(unsigned j=0;j<power;j++) amp=amp*sh.n;
                return amp*sh.P*I(2);
            });
            report<<"mass_"<<power<<" "<<mass.value<<"\n";
            report.flush();
        }
        ref.build_filters(prefix,argc==8?std::filesystem::path(argv[7]).parent_path().string()+"/":"");
        for(unsigned j=0;j<ref.filters.size();j++) {
            ref.filters[j]->save(prefix+"filter-"+std::to_string(j)+".txt");
            Integral half=integrate([&](const I &t,unsigned n) {
                return ref.local_filter_left_jet(t,n,int(j))*ref.local_shape(t,n,true).P;
            });
            I H=I(2)*half.value+log(I(1)+ref.filters[j]->range(I(1)));
            report<<"H_"<<j<<" "<<H<<"\n";
            report<<"H_signal_"<<j<<" "<<ref.target_H(ref.par.scales[j])-H<<"\n";
            report.flush();
        }
        ref.linear_filter->save(prefix+"linear-filter.txt");
        Integral S=integrate([&](const I &t,unsigned n) {
            Jet a=ref.local_filter_left_jet(t,n,-1);
            return square(a)*ref.local_shape(t,n,true).P*I(2);
        });
        // Exact reflection identity: all exterior tails and the cross-half
        // interaction combine to A(1)^2. No numerical tail cutoff is used.
        I second=S.value+square(ref.linear_filter->range(I(1)));
        report<<"S "<<second<<"\n"<<"S_signal "<<square(ref.par.k)/I(6)-second<<"\n";
        if(ref.par.beta.zero()) {
            // For a symmetric density, ordering the three variables in
            // tr(K_m^3) gives R=(k^3/24)tr(K_m^3)
            // =(k/2)*integral_left m(x) A(x) A(-x) dx. This uses only the
            // already verified linear filter; there is no positive scale.
            Integral third=integrate([&](const I &t,unsigned n) {
                Shape sh=ref.local_shape(t,n,true);
                Jet left=ref.local_filter_left_jet(t,n,-1);
                Jet right=reflected(ref.linear_filter->jet(I(2)-t,n));
                return sh.P*square(sh.n)*left*right*(ref.par.k/I(2));
            });
            report<<"R "<<third.value<<"\nR_signal "<<pow(ref.par.k,3)/I(45)-third.value<<"\n";
        }
        finish_output(report);
        ref.build_complements(prefix,argc==8?std::filesystem::path(argv[7]).parent_path().string()+"/":"");
        check_runtime();
        std::cout<<"reference verification complete\n";
        cert::finish_trace(); return 0;
    } catch(const std::exception &e) {
        std::cerr<<"REJECT: "<<e.what()<<"\n"; return 1;
    }
}
