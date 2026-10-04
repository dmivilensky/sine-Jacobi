// Enclose the series of lem:sine-sum and the computed value ca:sine.
#include "trace.hpp"
#include <fstream>
#include <filesystem>
#include <limits>

using namespace cert;
using Polynomial=std::vector<I>;

static Polynomial multiply(const Polynomial &a,const Polynomial &b) {
    Polynomial c(a.size()+b.size()-1);
    for(std::size_t i=0;i<a.size();i++)for(std::size_t j=0;j<b.size();j++)
        c[i+j]=c[i+j]+a[i]*b[j];
    return c;
}
static Polynomial power(Polynomial a,unsigned n) {
    Polynomial p{I(1)};
    while(n){if(n&1)p=multiply(p,a);n>>=1;if(n)a=multiply(a,a);}
    return p;
}

/* The real theta identity is checked in the published FGS paper,
 * Monatshefte fuer Mathematik 206 (2025), pp.566-567, (4.6),(4.8),
 * DOI 10.1007/s00605-024-02051-0. The article proves the real nome
 * inversion and the exponential integral/tail estimate used below.
 * This computation needs no quadrature rule and no cutoff in space.
 */
int main(int argc,char **argv) {
    try {
        require(argc==4,"usage: verify_sine bits absolute_error output_directory");
        precision=cert::count_argument(argv[1]);require(precision>=64,"precision too low");
        I tolerance(argv[2]);require(tolerance.positive(),"positive accuracy required");
        check_runtime();
        std::string out=argv[3];std::filesystem::create_directories(out);if(out.back()!='/')out+='/';
        I p=pi(),a=exp(-p),MA=inverse(I(1)-square(a));
        I MB=I(1)+I(2)*a/(I(1)-pow(a,3));
        auto integral=[&](const I &lambda) {
            // Integral_pi^infinity s^2 exp(-lambda*s) ds, by two
            // integrations by parts. All factors are positive.
            return exp(-lambda*p)*(square(p)/lambda+I(2)*p/square(lambda)
                                       +I(2)/pow(lambda,3));
        };
        I factor=I(512)/pow(p,3),tail;
        unsigned degree=0;
        for(;;++degree) {
            require(degree+2<=unsigned(std::numeric_limits<int>::max()/12)/(degree+2),
                    "series dimension exceeds this build's integer indexing");
            I ca(int((degree+1)*(degree+2))),cb(int((degree+1)*(degree+1)));
            I ra=integral(I(3)/I(2)+ca)/(I(1)-exp(-I(int(2*(degree+2)))*p));
            I rb=integral(I(3)/I(2)+cb)/(I(1)-exp(-I(int(2*degree+3))*p));
            tail=upper(factor*(I(6)*pow(MA,5)*pow(MB,6)*ra
                              +I(12)*pow(MA,6)*pow(MB,5)*rb));
            numerical_trace()<<"truncation "<<degree<<" ra "<<ra<<" rb "<<rb<<" tail "<<tail<<"\n";
            if(mpfr_cmp(tail.hi.x,tolerance.lo.x)<=0)break;
        }
        Polynomial A(degree*(degree+1)+1),B(degree*degree+1);
        for(unsigned j=0;j<=degree;j++)A[j*(j+1)]=I(1);
        B[0]=I(1);
        for(unsigned j=1;j<=degree;j++)B[j*j]=I(j&1?-2:2);
        Polynomial C=multiply(power(A,6),power(B,6));
        I value;
        auto polynomial=cert::output_file(out+"sine-polynomial.txt");
        polynomial<<"SINE_POLYNOMIAL 1\n"<<degree<<" "<<C.size()<<"\n";
        for(unsigned j=0;j<C.size();j++) {
            require(width(C[j]).zero(),"precision insufficient for exact integer coefficients");
            polynomial<<j<<" "<<C[j]<<"\n";
            I term_integral=integral(I(int(j))+I(3)/I(2));
            I term=C[j]*term_integral;
            value=value+term;
            numerical_trace()<<"term "<<j<<" coefficient "<<C[j]<<" integral "<<term_integral<<" contribution "<<term<<" sum "<<value<<"\n";
        }
        value=value*factor;
        I energy=value+centered(tail);
        auto report=cert::output_file(out+"sine-values.txt");
        report<<"SINE_VALUES 1\n"<<"energy "<<energy<<"\n"
              <<"finite_sum "<<value<<"\ntail "<<tail<<"\n"
              <<"gain_upper "<<I(2)*sqrt(energy)<<"\n";
        finish_output(polynomial);finish_output(report);
        check_runtime();
        std::cout<<"theta truncation degree "<<degree<<"\nenergy "<<energy<<"\n";
        cert::finish_trace(); return 0;
    }catch(const std::exception &e){std::cerr<<"REJECT: "<<e.what()<<"\n";return 1;}
}
