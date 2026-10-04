// Recompute flow/cover inequalities (lem:regular-criterion, lem:box-bound)
// and prices at a common scale, justified by convexity in lem:conjugate.
#include "state_cover.hpp"
#include "compact_cover.hpp"
#include <filesystem>
#include <memory>

using namespace transport;

struct Tree {
    bool leaf=false;
    int axis=-1;
    std::array<std::unique_ptr<Tree>,2> child;
    void insert(const std::string &path,std::size_t index=0) {
        cert::require(!leaf,"duplicate or overlapping cover path");
        if(index==path.size()) {
            cert::require(axis<0,"cover leaf contains an earlier leaf");
            leaf=true;return;
        }
        char c=path[index];
        cert::require(c>='0'&&c<='3',"invalid cover path");
        int direction=c-'0',which_axis=direction/2,which_child=direction%2;
        cert::require(axis<0||axis==which_axis,"inconsistent subdivision axes");
        axis=which_axis;
        if(!child[which_child])child[which_child]=std::make_unique<Tree>();
        child[which_child]->insert(path,index+1);
    }
    bool complete() const {
        return leaf||(axis>=0&&child[0]&&child[1]&&child[0]->complete()&&child[1]->complete());
    }
};
static double coordinate(std::istream &in) {
    std::string s;in>>s;cert::require(bool(in),"missing finite binary coordinate");
    std::size_t used=0;double x=std::stod(s,&used);
    cert::require(used==s.size()&&std::isfinite(x),"invalid binary coordinate");
    // Producer outputs binary64 hexadecimal constants. Requiring that
    // format makes each input an exact dyadic, not an unspecified decimal.
    std::string absolute=s[0]=='-'?s.substr(1):s;
    cert::require(absolute.rfind("0x",0)==0&&absolute.find('p')!=std::string::npos,
                  "cover coordinate must be hexadecimal");
    cert::I exact(s);
    cert::require(cert::contained(exact,I(x))&&cert::contained(I(x),exact),
                  "cover coordinate is not exactly binary64 representable");
    return x;
}

int main(int argc,char **argv) {
    try {
        cert::require(argc==10||argc==13,
          "usage: verify_transport reference polynomial cache bits order flow_tol quadrature_tol cover report [half left right]");
        cert::precision=cert::count_argument(argv[4]);cert::require(cert::precision>=64,"precision too low");
        unsigned order=cert::count_argument(argv[5]);I flow_tol(argv[6]),quad_tol(argv[7]);
        cert::check_runtime();boxes::check_runtime();
        bool partial=argc==13;
        unsigned selected_half=0;
        double begin=0,end=1,verified_left=0;
        if(partial) {
            selected_half=cert::count_argument(argv[10]);
            std::istringstream bounds(std::string(argv[11])+" "+argv[12]);
            begin=coordinate(bounds);end=coordinate(bounds);cert::require_end(bounds);
            cert::require(selected_half<2&&0<=begin&&begin<end&&end<=1,"invalid verification region");
            verified_left=begin;
        }
        // Every worker independently proves all prerequisite flow predicates.
        // Only the state inequalities and price quadratures are restricted
        // to its explicitly reported closed spatial region.
        std::string cache=argv[3];if(cache.back()!='/')cache+='/';
        cert::Reference ref(cert::Parameters(argv[1]),order,flow_tol,cache+"support-filter-flow.txt");
        ref.build_filters("",cache);
        if(std::filesystem::exists(cache+"complement-0.txt"))ref.build_complements("",cache);
        Parameters par(argv[2]);
        cert::require(ref.par.scales.size()==2,"two profile scales required");
        for(unsigned j=0;j<2;j++)
            cert::require(convert(ref.par.scales[j]).lo==par.scales[j].lo
                          &&convert(ref.par.scales[j]).hi==par.scales[j].hi,"filter scale mismatch");
        CoverInput source(argv[8]);auto &input=source.stream();
        std::string magic;input>>magic;unsigned version=cert::read_count(input);
        cert::require(magic=="TRANSPORT_COVER"&&(version==1||version==2),"invalid cover format");
        unsigned expected_half=0;
        double expected_left=0;
        std::size_t cells=0,input_leaves=0,verified_cells=0,verified_leaves=0;
        bool complete=false;
        I total;
        double minimum_dilation=1,final_dilation=1;
        for(;;) {
            std::string tag;input>>tag;
            cert::require(bool(input),"cover is truncated before its completion record");
            if(tag=="complete") {
                std::size_t count=cert::read_count(input),leaf_count=cert::read_count(input);
                if(version==2)final_dilation=coordinate(input);
                cert::require(final_dilation>0&&final_dilation<=minimum_dilation,
                              "final homothety exceeds an accepted cell scale");
                cert::require(expected_half==2&&expected_left==0&&count==cells&&leaf_count==input_leaves,
                              "cover does not exhaust both closed spatial halves");
                cert::require(bool(input),"truncated completion record");
                input>>std::ws;cert::require(input.eof(),"unexpected records after completion");
                complete=true;
                break;
            }
            cert::require(tag=="cell","unexpected cover record");
            unsigned half=cert::read_count(input);
            double left=coordinate(input),right=coordinate(input);
            double dilation=version==2?coordinate(input):1.;
            cert::require(dilation>0&&dilation<=1,"invalid cell homothety");
            minimum_dilation=std::min(minimum_dilation,dilation);par.dilation=D(dilation);
            double intercept=coordinate(input),slope=coordinate(input);
            std::size_t count=cert::read_count(input);
            cert::require(half==expected_half&&left==expected_left&&left<right&&right<=1&&count>=2,
                          "spatial cells are not a complete ordered partition");
            bool selected=!partial||(half==selected_half&&left<end&&right>begin);
            if(partial&&selected) {
                cert::require(left==verified_left&&right<=end,
                              "spatial cell crosses a verification-region boundary");
                verified_left=right;
            }
            double mid=cell_center(left,right);
            std::unique_ptr<Profile> whole,middle;
            std::unique_ptr<Cover> checker;
            if(selected) {
                whole=std::make_unique<Profile>(ref,cert::hull(I(left),I(right)),half);
                middle=std::make_unique<Profile>(ref,I(mid),half);
                checker=std::make_unique<Cover>(par,*whole,*middle);
            }
            Price price{mid,intercept,slope};
            std::array<Tree,2> trees;
            for(std::size_t j=0;j<count;j++) {
                unsigned branch=cert::read_count(input);std::string path;input>>path;
                if(!selected) {
                    // These leaves are proved by the worker whose reported
                    // region contains this cell. Their values do not enter
                    // this partial theorem or its integral enclosure.
                    std::string upper;input>>upper;
                    cert::require(bool(input),"truncated state leaf outside verification region");
                    continue;
                }
                double recorded_upper=coordinate(input);
                cert::require(branch<=1&&recorded_upper<0,"invalid leaf branch or nonnegative upper bound");
                if(path=="-")path.clear();
                trees[branch].insert(path);
                auto b=checker->bound(decode_rectangle(branch,path),price);
                cert::require(b.upper<=recorded_upper,"recomputed leaf bound exceeds witness");
                cert::numerical_trace()<<"state "<<half<<" "<<left<<" "<<right
                    <<" "<<branch<<" "<<(path.empty()?"-":path)<<" upper "<<b.upper<<"\n";
            }
            if(selected)cert::require(trees[0].complete()&&trees[1].complete(),"conditional state cover has a hole");
            std::string integral_tag;I recorded;input>>integral_tag>>recorded;
            cert::require(integral_tag=="integral"&&bool(input),"missing price integral");
            if(selected) {
                auto integral=cert::quadrature([&](const I &t,unsigned n) {
                    auto sh=ref.local_shape(t,n,true);
                    Jet e=Jet(n,I(intercept))+(cert::variable(t,n)-Jet(n,I(mid)))*I(slope);
                    return cert::cube(sh.t)*sh.z*e*(I(3)/ref.par.k);
                },left,right,order,quad_tol);
                cert::require(cert::contained(integral.value,recorded),"price integral does not enclose recomputation");
                total=total+integral.value/I(dilation);
                ++verified_cells;verified_leaves+=count;
                if((verified_cells&(verified_cells-1))==0)
                    std::cout<<"rechecked spatial cells "<<verified_cells<<" state leaves "<<verified_leaves<<"\n"<<std::flush;
            }
            ++cells;input_leaves+=count;expected_left=right;
            if(right==1){++expected_half;expected_left=0;}
        }
        cert::require(complete&&verified_cells>0&&(!partial||verified_left==end),
                      "incomplete transport verification domain");
        std::string report_path=argv[9];
        auto report=cert::output_file(report_path);
        if(partial)report<<"VERIFIED_TRANSPORT_PART 1\ndomain "<<selected_half<<" "
                         <<std::hexfloat<<begin<<" "<<end<<"\n";
        else report<<"VERIFIED_TRANSPORT 2\n";
        report<<"dilation "<<I(final_dilation)
                  <<"\nprice "<<total*I(final_dilation)<<"\nundilated_price "<<total
                  <<"\ncells "<<verified_cells<<"\nleaves "<<verified_leaves<<"\n";
        cert::finish_output(report);
        cert::check_runtime();boxes::check_runtime();
        std::cout<<(partial?"accepted: stated closed region, all its state inequalities and integrals recomputed\n"
                           :"accepted: complete spatial and state covers, all inequalities and integrals recomputed\n");
        cert::finish_trace(); return 0;
    } catch(const std::exception &e) {
        std::cerr<<"REJECT: "<<e.what()<<"\n";return 1;
    }
}
