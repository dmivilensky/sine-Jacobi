#pragma once
// Decode one leaf at a time into the existing acceptance reader. No expanded
// text cover is written to disk; the interval/state predicates are unchanged.
#include "interval.hpp"
#include <fstream>
#include <sstream>
#include <streambuf>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstring>
#include <limits>

namespace transport {
class PackedCoverWriter {
    std::ofstream file;
    void integer(std::uint64_t n,unsigned bytes) {
        for(unsigned i=0;i<bytes;i++)file.put(char((n>>(8*i))&255));
    }
    void number(double x) {
        cert::require(std::isfinite(x),"cannot save nonfinite cover value");
        static_assert(sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"IEEE binary64 required");
        std::uint64_t bits;std::memcpy(&bits,&x,8);integer(bits,8);
    }
    struct Node {
        bool leaf=false;int axis=-1;
        std::unique_ptr<Node> child[2];
        void insert(const std::string &word) {
            Node *node=this;unsigned du=0,dv=0;
            for(char c:word) {
                cert::require(c>='0'&&c<='3',"invalid cover word");
                unsigned a=(c-'0')/2,b=(c-'0')%2;du+=a==0;dv+=a==1;
                cert::require(du<=52&&dv<=52,"cover depth exceeds exact contract");
                cert::require(!node->leaf&&(node->axis<0||node->axis==int(a)),"inconsistent cover tree");
                node->axis=a;
                if(!node->child[b])node->child[b]=std::make_unique<Node>();
                node=node->child[b].get();
            }
            cert::require(!node->leaf&&node->axis<0,"overlapping cover leaf");node->leaf=true;
        }
    };
public:
    PackedCoverWriter(const std::string &path,bool partial,unsigned half,double begin,double end)
        :file(path,std::ios::binary) {
        cert::require(bool(file),"cannot write packed cover");
        file.exceptions(std::ios::badbit|std::ios::failbit);
        file.write("SJCV1\r\n\0",8);integer(partial?1:0,1);integer(partial?half:0,1);
        number(partial?begin:0.);number(partial?end:1.);
    }
    template<class Leaves>
    void cell(unsigned half,double left,double right,double dilation,double intercept,double slope,
              double upper,const Leaves &leaves,const cert::I &integral) {
        cert::require(leaves.size()>=2&&leaves.size()<=(1u<<24)&&std::isfinite(upper)&&upper<0,
                      "invalid packed cover cell");
        Node roots[2];
        for(const auto &leaf:leaves) {
            cert::require(leaf.box.branch<2&&leaf.upper<=upper,"invalid common state margin");
            roots[leaf.box.branch].insert(leaf.box.path);
        }
        std::vector<const Node*> stack{&roots[1],&roots[0]};
        std::vector<unsigned char> packed((2*leaves.size()+1)/4,0);std::size_t index=0;
        while(!stack.empty()) {
            const Node *node=stack.back();stack.pop_back();
            cert::require(index<2*leaves.size()-2,"cover node count overflow");
            if(!node->leaf) {
                cert::require(node->axis>=0&&node->child[0]&&node->child[1],"cover tree has a hole");
                packed[index/4]|=static_cast<unsigned char>((node->axis+1)<<(2*(index%4)));
                stack.push_back(node->child[1].get());stack.push_back(node->child[0].get());
            }
            ++index;
        }
        cert::require(index==2*leaves.size()-2,"cover node count mismatch");
        std::ostringstream encoded;encoded<<integral;auto price=encoded.str();
        cert::require(price.size()>0&&price.size()<=4096,"price encoding too long");
        file.put('C');integer(half,1);
        for(double x:{left,right,dilation,intercept,slope,upper})number(x);
        integer(leaves.size(),8);integer(price.size(),4);
        file.write(reinterpret_cast<const char*>(packed.data()),packed.size());
        file.write(price.data(),price.size());file.flush();
    }
    void finish(std::size_t cells,std::size_t leaves,double dilation) {
        file.put('E');integer(cells,8);integer(leaves,8);number(dilation);file.flush();file.close();
    }
};
class PackedCoverBuffer : public std::streambuf {
    std::ifstream file;
    std::string buffer,integral;
    std::vector<unsigned char> tree;
    struct Pending {unsigned branch,du,dv;std::string word;};
    std::vector<Pending> pending;
    std::uint64_t leaves=0,nodes=0,visited=0,emitted=0;
    double upper=0;
    int phase=0;
    std::uint64_t integer(unsigned bytes) {
        std::uint64_t v=0;
        for(unsigned i=0;i<bytes;i++) {
            int c=file.get();cert::require(c!=EOF,"truncated packed cover");
            v|=std::uint64_t(static_cast<unsigned char>(c))<<(8*i);
        }
        return v;
    }
    double number() {
        static_assert(sizeof(double)==8&&std::numeric_limits<double>::is_iec559,
                      "IEEE binary64 required");
        std::uint64_t bits=integer(8);double v;std::memcpy(&v,&bits,8);
        cert::require(std::isfinite(v),"nonfinite packed coordinate");return v;
    }
    std::string bytes(std::size_t n) {
        std::string s(n,'\0');file.read(s.data(),n);
        cert::require(std::size_t(file.gcount())==n,"truncated packed payload");return s;
    }
    bool next() {
        std::ostringstream out;out<<std::hexfloat;
        if(phase==0) {
            cert::require(bytes(8)==std::string("SJCV1\r\n\0",8),"packed cover magic");
            auto mode=integer(1),half=integer(1);double begin=number(),end=number();
            cert::require(mode==0&&half==0&&begin==0&&end==1,"full packed cover required");
            phase=1;buffer="TRANSPORT_COVER 2\n";return true;
        }
        if(phase==3) {
            cert::require(file.get()==EOF,"trailing packed cover bytes");return false;
        }
        if(phase==2) {
            while(!pending.empty()) {
                cert::require(visited<nodes,"truncated tree structure");
                Pending p=std::move(pending.back());pending.pop_back();
                unsigned tag=(tree[visited/4]>>(2*(visited%4)))&3;++visited;
                cert::require(tag<3,"reserved packed tree tag");
                if(tag==0) {
                    ++emitted;
                    out<<p.branch<<" "<<(p.word.empty()?"-":p.word)<<" "<<upper<<"\n";
                    buffer=out.str();return true;
                }
                p.du+=tag==1;p.dv+=tag==2;
                cert::require(p.du<=52&&p.dv<=52,"packed state depth exceeds exact contract");
                Pending right=p;right.word+=char('0'+2*(tag-1)+1);
                p.word+=char('0'+2*(tag-1));
                pending.push_back(std::move(right));pending.push_back(std::move(p));
            }
            cert::require(visited==nodes&&emitted==leaves,"packed tree count mismatch");
            cert::require(nodes%4==0||(tree.back()>>(2*(nodes%4)))==0,"nonzero tree padding");
            buffer="integral "+integral+"\n";phase=1;return true;
        }
        int tag=file.get();cert::require(tag!=EOF,"missing packed completion");
        if(tag=='E') {
            auto cells=integer(8),count=integer(8);double dilation=number();
            out<<"complete "<<cells<<" "<<count<<" "<<dilation<<"\n";
            buffer=out.str();phase=3;return true;
        }
        cert::require(tag=='C',"unknown packed cover record");
        auto half=integer(1);
        double left=number(),right=number(),dilation=number(),intercept=number(),slope=number();
        upper=number();leaves=integer(8);auto length=integer(4);
        cert::require(leaves>=2&&leaves<=(1u<<24)&&upper<0,"invalid packed leaf count or margin");
        cert::require(length>0&&length<=4096,"invalid packed integral length");
        nodes=2*leaves-2;
        auto raw=bytes((nodes+3)/4);tree.assign(raw.begin(),raw.end());
        integral=bytes(length);
        // Prevent record injection through the only variable text field.
        std::istringstream check(integral);cert::I price;check>>price;
        cert::require(bool(check),"invalid packed price interval");cert::require_end(check);
        pending={{1,0,0,""},{0,0,0,""}};visited=emitted=0;phase=2;
        out<<"cell "<<half<<" "<<left<<" "<<right<<" "<<dilation<<" "
           <<intercept<<" "<<slope<<" "<<leaves<<"\n";
        buffer=out.str();return true;
    }
protected:
    int_type underflow() override {
        if(gptr()&&gptr()<egptr())return traits_type::to_int_type(*gptr());
        if(!next())return traits_type::eof();
        setg(buffer.data(),buffer.data(),buffer.data()+buffer.size());
        return traits_type::to_int_type(*gptr());
    }
public:
    explicit PackedCoverBuffer(const std::string &path):file(path,std::ios::binary) {
        cert::require(bool(file),"cannot open packed cover");
    }
};
class CoverInput {
    std::ifstream file;
    std::unique_ptr<PackedCoverBuffer> packed;
    std::unique_ptr<std::istream> decoded;
public:
    explicit CoverInput(const std::string &path):file(path,std::ios::binary) {
        cert::require(bool(file),"cannot read cover");
        char magic[8]{};file.read(magic,8);file.clear();file.seekg(0);
        if(std::memcmp(magic,"SJCV1\r\n\0",8)==0) {
            packed=std::make_unique<PackedCoverBuffer>(path);
            decoded=std::make_unique<std::istream>(packed.get());
            decoded->exceptions(std::ios::badbit);
        }
    }
    std::istream &stream() {return decoded?*decoded:static_cast<std::istream &>(file);}
};
} // namespace transport
