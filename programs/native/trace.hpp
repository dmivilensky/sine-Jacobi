#pragma once
// Trace policy is observational: saved flows and acceptance predicates are unchanged.
#include "series.hpp"
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <map>
#include <streambuf>

namespace cert {
struct DiscardTrace : std::streambuf {
    unsigned long long bytes=0;
    std::streamsize xsputn(const char *,std::streamsize n) override {bytes+=n;return n;}
    int overflow(int c) override {
        if(!traits_type::eq_int_type(c,traits_type::eof()))++bytes;
        return traits_type::not_eof(c);
    }
};
struct TraceState {
    std::ofstream file;
    DiscardTrace buffer;
    std::ostream sink{&buffer};
    bool full=false;
    unsigned long long requests=0;
    std::map<std::string,unsigned long long> cells;
    TraceState() {
        const char *path=std::getenv("SINEJACOBI_TRACE");
        require(path&&*path,"a numerical trace path is required");
        const char *mode=std::getenv("SINEJACOBI_TRACE_MODE");
        require(!mode||std::strcmp(mode,"summary")==0||std::strcmp(mode,"full")==0,"invalid trace mode");
        full=mode&&std::strcmp(mode,"full")==0;
        file=output_file(path); file<<std::hexfloat; sink<<std::hexfloat;
        if(!full)file<<"NUMERICAL_TRACE_SUMMARY 1\nmode summary\n";
    }
};
inline TraceState &trace_state() {static TraceState state;return state;}
inline std::ostream &numerical_trace() {
    auto &s=trace_state();++s.requests;
    return s.full?static_cast<std::ostream &>(s.file):s.sink;
}
inline void finish_trace() {
    auto &s=trace_state();
    if(!s.full) {
        s.file<<"stream_requests "<<s.requests<<"\ndiscarded_stream_bytes "<<s.buffer.bytes<<"\n";
        for(const auto &v:s.cells)s.file<<"cells "<<v.first<<" "<<v.second<<"\n";
        s.file<<"complete\n";
    }
    finish_output(s.file);
}
inline void trace_cell(const char *kind,double left,double right,
                       const Jet &center,const Jet &whole,const I &value) {
    auto &s=trace_state();++s.cells[kind];
    // In full mode operator<< checked every interval before formatting.
    // Preserve those rejection checks when derivative text is suppressed.
    value.check();
    for(const auto &x:center.c)x.check();
    for(const auto &x:whole.c)x.check();
    if(!s.full)return;
    auto &out=numerical_trace();
    out<<kind<<" "<<left<<" "<<right<<" value "<<value<<"\ncenter";
    for(const auto &x:center.c)out<<" "<<x;
    out<<"\nwhole";
    for(const auto &x:whole.c)out<<" "<<x;
    out<<"\n";
    require(bool(out),"cannot write numerical trace");
}
} // namespace cert
