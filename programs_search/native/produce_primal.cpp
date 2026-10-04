// Produce the dyadic center and cube for lem:contraction; retain every Newton step.
#include "primal.hpp"
using namespace cert;

int main(int argc,char **argv) {
    try {
        require(argc==8,"usage: produce_primal proposal bits order flow_tol quad_tol radius_goal output");
        precision=cert::count_argument(argv[2]);require(precision>=64,"precision too low");
        unsigned order=cert::count_argument(argv[3]);I ft(argv[4]),qt(argv[5]),goal(argv[6]);
        require(goal.positive(),"positive radius goal required");
        check_runtime();PrimalParameters parameters(argv[1]);parameters.make_dyadic();
        std::string output=argv[7];std::filesystem::create_directories(output);
        std::unique_ptr<PrimalEvaluation> center;I rho;
        for(long iteration=0;iteration<precision;iteration++) {
            center=std::make_unique<PrimalEvaluation>(parameters,order,ft,qt);
            std::string directory=output+"/iteration-"+std::to_string(iteration);
            center->build_flows(directory,false);center->integrate();
            parameters.save(directory+"/profile.txt");
            save_primal_partition(directory+"/partition.txt",center->partition);
            auto correction=center->correction();I norm=infinity_bound(correction);
            numerical_trace()<<"Newton "<<iteration<<" correction";
            for(const auto &v:correction)numerical_trace()<<" "<<v;
            numerical_trace()<<" norm "<<norm<<"\n"<<std::flush;
            std::cout<<"iteration "<<iteration<<" energy "<<center->integrals[2]
                     <<" correction "<<norm<<" cells "<<center->partition.size()<<"\n"<<std::flush;
            if(mpfr_cmp((I(2)*norm).hi.x,goal.lo.x)<0) {
                // The positive arithmetic-resolution term also handles an
                // exactly zero center residual without a zero-radius cube.
                rho=upper(I(2)*norm+inverse(pow(I(2),unsigned(precision))));
                std::filesystem::rename(directory,output+"/central");break;
            }
            bool changed=false;
            for(unsigned j=0;j<correction.size();j++) {
                unsigned index=parameters.directions[j];
                I full=midpoint(parameters.coefficients[index]-midpoint(correction[j]));
                I next(midpoint(full).lower_double());
                if(contained(next,parameters.coefficients[index]))next=full;
                changed=changed||!contained(next,parameters.coefficients[index]);
                parameters.coefficients[index]=next;
            }
            require(changed,"primal correction stagnated before the requested radius");
            require(iteration+1<precision,"primal correction iteration limit");
        }
        require(center&&rho.positive(),"primal center was not produced");
        parameters.save(output+"/profile.txt");
        save_primal_partition(output+"/partition.txt",center->partition);
        auto radius_file=cert::output_file(output+"/radius.txt");radius_file<<rho<<"\n";radius_file.close();
        PrimalParameters perturbed=parameters;
        for(unsigned index:perturbed.directions)perturbed.coefficients[index]=perturbed.coefficients[index]+centered(rho);
        PrimalEvaluation cube(perturbed,order,ft,qt);
        cube.build_flows(output+"/cube",false);cube.integrate(center->partition);
        report_primal(output+"/primal-values.txt",*center,cube,rho);
        cert::finish_trace(); return 0;
    } catch(const std::exception &e) {std::cerr<<"REJECT: "<<e.what()<<"\n";return 1;}
}
