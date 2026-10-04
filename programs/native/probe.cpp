#include "state_cover.hpp"
#include <cfloat>
int main() {
    try {
        cert::precision=160;cert::check_runtime();boxes::check_runtime();
        std::cout<<"MPFR "<<mpfr_get_version()<<" GMP "<<gmp_version
                 <<" binary64_digits "<<DBL_MANT_DIG<<" C++ "<<__cplusplus<<"\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
