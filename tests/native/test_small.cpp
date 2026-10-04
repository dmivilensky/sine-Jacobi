// Analytically solvable, bounded tests. Never runs a full-sized flow.
#include "state_cover.hpp"
#include "primal.hpp"
#include <filesystem>
using namespace cert;
static void overlap(boxes::D a,boxes::D b) {
    require(std::max(a.lo,b.lo)<=std::min(a.hi,b.hi),"direct/regrouped enclosures disagree");
}
static void overlap(const boxes::A &a,const boxes::A &b) {
    overlap(a.v,b.v);
    for(unsigned i=0;i<3;i++) {
        overlap(a.g[i],b.g[i]);
        for(unsigned j=0;j<3;j++)overlap(a.h[i][j],b.h[i][j]);
    }
}
int main(int argc,char **argv) {
    try {
        require(argc==3,"usage: test_small work_directory transport_parameters");
        precision=160;check_runtime();boxes::check_runtime();
        std::filesystem::create_directories(argv[1]);
        require(contained(I(0),square(hull(I(-2),I(3)))),"square crosses zero");
        bool rejected=false;
        try { (void)inverse(hull(I(-1),I(1))); } catch(const std::runtime_error &) {rejected=true;}
        require(rejected,"division through zero was accepted");
        Jet q=cube(variable(I(2),4));
        require(contained(I(8),q[0])&&contained(I(12),q[1])&&contained(I(6),q[2])&&contained(I(1),q[3]),"normalized jet coefficients");
        CubicQuotient quotient([](const I &t,unsigned n){return cube(variable(t,n));},8,.125);
        auto result=quotient.evaluate(hull(I(0),I(.125)),3);
        require(contained(I(1),result[0])&&result[1].zero()&&result[2].zero(),"removable cubic quotient");
        Provider ode=[](const I &,unsigned n){return QuadraticODE{Jet(n,I(1)),Jet(n,I(-2)),Jet(n)};};
        Flow f(ode,8,I("1e-16"));f.segment(0,.125,I(0),.125);
        I target=(I(1)-exp(I(-.25)))/I(2);
        require(contained(target,f.endpoint),"analytic stable ODE value not enclosed");
        std::string file=std::string(argv[1])+"/flow.txt";f.save(file);
        Flow replay(ode,8,I("1e-16"));replay.load_and_verify(file,0,.125,I(0));
        require(contained(target,replay.endpoint),"saved-flow replay failed");
        auto integral=quadrature([](const I &t,unsigned n){return square(variable(t,n));},0,1,4,I("1e-14"));
        require(contained(I(0),I(3)*integral.value-I(1)),"polynomial integral not enclosed");
        transport::Parameters par(argv[2]);
        for(unsigned half=0;half<2;half++)for(double time:{0.,.2,.5,1.}) {
            using boxes::A;using boxes::D;
            transport::Profile p;p.side=half;p.k=transport::convert(square(pi())/I(4));
            p.t=boxes::variable(D(time),2);p.q=half?A(2)-boxes::pow(p.t,3):boxes::pow(p.t,3);
            p.z=A(D(1.1));p.drift=A(D(1.3));p.ell=p.t*(A(2)-boxes::pow(p.t,3))/(boxes::square(p.z)*p.k);
            p.right={A(D(.2)),A(D(.4))};p.left={A(D(.3)),A(D(.5))};
            p.right_linear=A(D(.7));p.left_moment=A(D(.4));
            transport::StateKernel kernel(par,p);
            for(double u:{0.,.3,.8,1.})for(double v:{0.,.3,.8,1.})for(unsigned branch=0;branch<2;branch++) {
                auto direct=transport::StateFormula::build(par,p,D(u),D(v),branch);
                auto grouped=kernel.evaluate(D(u),D(v),branch);
                overlap(direct.b,grouped.b);overlap(direct.c,grouped.c);overlap(direct.ell,grouped.ell);
            }
        }
        // Trace suppression must retain checks formerly performed by I<<.
        Jet invalid(0);mpfr_set_nan(invalid[0].lo.x);rejected=false;
        try {trace_cell("invalid",0,1,invalid,Jet(0),I(0));}
        catch(const std::runtime_error &) {rejected=true;}
        require(rejected,"trace mode lost interval validity checking");mpfr_clear_flags();
        check_runtime();finish_trace();
        std::cout<<"PASS: intervals, jets, cubic quotient, exact ODE replay, quadrature, 256 state-kernel comparisons\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
