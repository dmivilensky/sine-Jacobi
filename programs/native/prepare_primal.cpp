// Build witnesses for the fixed dyadic profile and fixed radius.
// No Newton step changes the defining coefficients or the preconditioner.
#include "primal.hpp"
using namespace cert;
int main(int argc,char **argv) {
    try {
        require(argc==8,"usage: prepare_primal profile radius bits order flow_tol quad_tol output");
        precision=count_argument(argv[3]); require(precision>=64,"precision too low");
        unsigned order=count_argument(argv[4]); I ft(argv[5]),qt(argv[6]); check_runtime();
        PrimalParameters parameters(argv[1]);
        std::ifstream input(argv[2]); I rho; input>>rho;
        require(bool(input)&&rho.positive()&&width(rho).zero(),"invalid fixed radius");
        require_end(input);
        std::string out=argv[7]; std::filesystem::create_directories(out);
        parameters.save(out+"/profile.txt");
        auto radius_file=output_file(out+"/radius.txt"); radius_file<<rho<<"\n"; finish_output(radius_file);
        PrimalEvaluation center(parameters,order,ft,qt);
        center.build_flows(out+"/central",false); center.integrate();
        parameters.save(out+"/central/profile.txt");
        save_primal_partition(out+"/partition.txt",center.partition);
        save_primal_partition(out+"/central/partition.txt",center.partition);
        for(unsigned j:parameters.directions) parameters.coefficients[j]=parameters.coefficients[j]+centered(rho);
        PrimalEvaluation cube(parameters,order,ft,qt);
        cube.build_flows(out+"/cube",false); cube.integrate(center.partition);
        report_primal(out+"/candidate-values.txt",center,cube,rho);
        finish_trace(); return 0;
    } catch(const std::exception &e) {std::cerr<<"REJECT: "<<e.what()<<"\n";return 1;}
}
