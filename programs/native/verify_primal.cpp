// Recompute the central and whole-cube flows and the lem:contraction predicates.
#include "primal.hpp"
using namespace cert;

int main(int argc,char **argv) {
    try {
        require(argc==7,"usage: verify_primal directory bits order flow_tol quad_tol output_values");
        std::string directory=argv[1];precision=cert::count_argument(argv[2]);
        require(precision>=64,"precision too low");unsigned order=cert::count_argument(argv[3]);
        I ft(argv[4]),qt(argv[5]);check_runtime();
        PrimalParameters parameters(directory+"/profile.txt");
        for(const I &c:parameters.coefficients)require(width(c).zero(),"central coefficient is not an exact dyadic");
        for(const auto &row:parameters.C)for(const I &c:row)require(width(c).zero(),"preconditioner is not exact");
        std::ifstream radius_file(directory+"/radius.txt");I rho;radius_file>>rho;
        std::string extra;require(bool(radius_file)&&!(radius_file>>extra)&&rho.positive()&&width(rho).zero(),
                                  "invalid primal radius record");
        auto partition=load_primal_partition(directory+"/partition.txt");
        PrimalEvaluation center(parameters,order,ft,qt);
        center.build_flows(directory+"/central",true);center.integrate(partition);
        for(unsigned index:parameters.directions)parameters.coefficients[index]=parameters.coefficients[index]+centered(rho);
        PrimalEvaluation cube(parameters,order,ft,qt);
        cube.build_flows(directory+"/cube",true);cube.integrate(partition);
        report_primal(argv[6],center,cube,rho);
        cert::finish_trace(); return 0;
    } catch(const std::exception &e) {std::cerr<<"REJECT: "<<e.what()<<"\n";return 1;}
}
