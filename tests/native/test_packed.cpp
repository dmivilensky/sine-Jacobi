#include "compact_cover.hpp"
#include "state_cover.hpp"
#include <filesystem>
using namespace cert;
int main(int argc,char **argv) {
    try {
        require(argc==3,"usage: test_packed mode path");precision=160;check_runtime();
        std::string mode=argv[1],path=argv[2];
        if(mode=="decode") {
            transport::CoverInput cover(path);std::string line;
            while(std::getline(cover.stream(),line))std::cout<<line<<"\n";
            require(cover.stream().eof(),"cover input failed");
        } else if(mode=="write") {
            transport::PackedCoverWriter out(path,false,0,0,1);
            std::vector<transport::Bound> bounds;
            for(unsigned branch=0;branch<2;branch++)for(unsigned child=0;child<2;child++) {
                transport::Bound b;b.box=transport::decode_rectangle(branch,std::to_string(2*branch+child));
                b.upper=-.125;bounds.push_back(b);
            }
            for(unsigned half=0;half<2;half++)out.cell(half,0,1,1,.25,0,-.0625,bounds,I(0));
            out.finish(2,8,1);
        } else if(mode=="state-replay") {
            std::string parameters=path+".parameters";
            {auto out=output_file(parameters);out<<"TRANSPORT_POLYNOMIAL 1\n2\n1\n2\n0\n0\n0\n0\n0\n";finish_output(out);}
            transport::Parameters par(parameters);
            transport::CoverInput cover(path);auto &in=cover.stream();std::string tag;unsigned version;
            in>>tag>>version;require(tag=="TRANSPORT_COVER"&&version==2,"test cover header");
            while(in>>tag&&tag=="cell") {
                unsigned half;std::string ls,rs,ds,is,ss;std::size_t count;
                in>>half>>ls>>rs>>ds>>is>>ss>>count;
                double l=std::stod(ls),r=std::stod(rs),mid=transport::cell_center(l,r);
                auto profile=[&](boxes::D t) {
                    using boxes::A;using boxes::D;
                    transport::Profile p;p.side=half;p.k=D(1);p.t=boxes::variable(t,2);
                    p.q=half?A(2)-boxes::pow(p.t,3):boxes::pow(p.t,3);
                    p.z=A(1);p.drift=A(1);p.ell=A(D(.2));
                    p.left={A(D(.1)),A(D(.1))};p.right=p.left;
                    p.left_moment=A(D(.1));p.right_linear=A(D(.1));return p;
                };
                auto whole=profile(boxes::D(l,r)),middle=profile(boxes::D(mid));
                transport::Cover checker(par,whole,middle);
                transport::Price price{mid,std::stod(is),std::stod(ss)};
                for(std::size_t i=0;i<count;i++) {
                    unsigned branch;std::string word,upper;in>>branch>>word>>upper;
                    auto result=checker.bound(transport::decode_rectangle(branch,word=="-"?"":word),price);
                    require(result.upper<=std::stod(upper),"recomputed state bound exceeds witness");
                }
                I value;in>>tag>>value;require(tag=="integral","test integral record");
            }
            require(tag=="complete","test completion");
            std::string rest;std::getline(in,rest);require_end(in);finish_trace();
        } else if(mode=="flow-write"||mode=="flow-replay") {
            Provider ode=[](const I &,unsigned n){return QuadraticODE{Jet(n,I(1)),Jet(n,I(-2)),Jet(n)};};
            Flow flow(ode,8,I("1e-16"));
            if(mode=="flow-write") {flow.segment(0,.125,I(0),.125);flow.save(path);}
            else flow.load_and_verify(path,0,.125,I(0));
            require(contained((I(1)-exp(I(-.25)))/I(2),flow.endpoint),"analytic ODE mismatch");
            finish_trace();
        } else throw std::runtime_error("unknown test mode");
        return 0;
    } catch(const std::exception &e) {std::cerr<<"REJECT: "<<e.what()<<"\n";return 1;}
}
