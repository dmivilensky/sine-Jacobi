#!/usr/bin/env python3
from __future__ import annotations
import math, json, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path('/mnt/data')
OUT=ROOT/'experiment_spectrum_adapted_v6'
FIGS=OUT/'figures'; TABLES=OUT/'tables'
if OUT.exists(): shutil.rmtree(OUT)
FIGS.mkdir(parents=True); TABLES.mkdir(parents=True)

COL={'block':'#24804c','single':'#315f9b','guard':'#c66a2d','merge':'#7254a2','gray':'#6a6a6a','black':'#202020','red':'#a94442','gold':'#c99a2e'}
plt.rcParams.update({
    'figure.dpi':190,'savefig.dpi':430,
    'font.size':9.3,'axes.labelsize':9.6,'axes.titlesize':10.2,'legend.fontsize':7.4,
    'xtick.labelsize':8.2,'ytick.labelsize':8.2,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.55,
    'mathtext.fontset':'dejavusans','lines.linewidth':1.65,
})

CONFIG={
    'base_blocks': [(1.0,1.1),(20.0,22.0),(400.0,440.0)],
    'guarded_blocks': [(0.95,2.05),(18.95,40.9),(381.1,458.9)],
    'guarded_hull': [(0.95,458.9)],
    'N_grid': list(range(9,76,3)),
    'N_main': 45,
    'ensemble_instances': 192,
    'eigs_per_block': 10,
    'residual_grid_per_block': 450,
    'leakage_deltas': list(np.linspace(0.0,0.16,33)),
    'leakage_instances': 128,
    'tol': 1e-2,
    'seed': 13062026,
}
(OUT/'spectrum_adapted_config.json').write_text(json.dumps(CONFIG,indent=2))

# ---------- helpers ----------
def save(fig,name):
    fig.savefig(FIGS/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIGS/f'{name}.png',dpi=190)
    plt.close(fig)

def panel(ax,letter):
    ax.text(-0.105,1.06,letter,transform=ax.transAxes,ha='left',va='top',fontsize=11,fontweight='bold')

def legend(ax,loc='best',**kw):
    ax.legend(loc=loc,frameon=True,framealpha=.96,edgecolor='none',handlelength=2.2,columnspacing=.9,**kw)

def log_cosh_n(n:int, y:float) -> float:
    a=math.acosh(max(float(y),1.0))
    return n*a - math.log(2.0) + math.log1p(math.exp(-2*n*a))

def cheb_T(n:int, x):
    x=np.asarray(x,float)
    out=np.empty_like(x,dtype=float)
    mid=np.abs(x)<=1
    out[mid]=np.cos(n*np.arccos(np.clip(x[mid],-1,1)))
    xp=x[~mid]
    if xp.size:
        vals=np.cosh(n*np.arccosh(np.abs(xp)))
        out[~mid]=np.where((xp<0)&(n%2==1),-vals,vals)
    return out

def cheb_residual_interval(lam,N,ell,u):
    lam=np.asarray(lam,float)
    z=(ell+u-2*lam)/(u-ell)
    eta=(ell+u)/(u-ell)
    return cheb_T(N,z)/cheb_T(N,np.array([eta]))[0]

def spectral_grid(blocks, per_block):
    xs=[]
    for ell,u in blocks:
        xs.append(np.linspace(ell,u,per_block))
    return np.concatenate(xs)

def block_params(blocks):
    ells=np.array([b[0] for b in blocks],float)
    us=np.array([b[1] for b in blocks],float)
    eta=(ells+us)/(us-ells)
    return ells,us,eta

def theta_dir(blocks,n):
    n=np.asarray(n,int); M=len(n)
    ells,us,eta=block_params(blocks)
    if M==1: return -log_cosh_n(n[0],eta[0])
    Zp=np.full(M,np.nan); Zm=np.full(M,np.nan)
    for a in range(M-1): Zp[a]=(2*us[-1]-ells[a]-us[a])/(us[a]-ells[a])
    for a in range(1,M): Zm[a]=(ells[a]+us[a]-2*ells[0])/(us[a]-ells[a])
    vals=[]
    for b in range(M):
        v=-log_cosh_n(int(n[b]),eta[b])
        for a in range(b): v += log_cosh_n(int(n[a]),Zp[a]) - log_cosh_n(int(n[a]),eta[a])
        for a in range(b+1,M): v += log_cosh_n(int(n[a]),Zm[a]) - log_cosh_n(int(n[a]),eta[a])
        vals.append(v)
    return float(max(vals))

def greedy_allocation(blocks,N):
    M=len(blocks)
    if M==1: return np.array([N],int), theta_dir(blocks,[N])
    n=np.ones(M,dtype=int)
    ells,us,eta=block_params(blocks)
    Zp=np.full(M,np.nan); Zm=np.full(M,np.nan)
    for a in range(M-1): Zp[a]=(2*us[-1]-ells[a]-us[a])/(us[a]-ells[a])
    for a in range(1,M): Zm[a]=(ells[a]+us[a]-2*ells[0])/(us[a]-ells[a])
    def scores(nv):
        S=[]
        for b in range(M):
            v=-log_cosh_n(int(nv[b]),eta[b])
            for a in range(b): v += log_cosh_n(int(nv[a]),Zp[a]) - log_cosh_n(int(nv[a]),eta[a])
            for a in range(b+1,M): v += log_cosh_n(int(nv[a]),Zm[a]) - log_cosh_n(int(nv[a]),eta[a])
            S.append(v)
        return np.array(S,float)
    for _ in range(M,N):
        best=None
        for c in range(M):
            nn=n.copy(); nn[c]+=1
            val=float(np.max(scores(nn)))
            if best is None or val<best[0]-1e-15 or (abs(val-best[0])<1e-15 and c<best[1]):
                best=(val,c)
        n[best[1]]+=1
    return n, float(np.max(scores(n)))

def block_residual(lam,blocks,n):
    lam=np.asarray(lam,float)
    res=np.ones_like(lam,dtype=float)
    for (ell,u),na in zip(blocks,n):
        z=(ell+u-2*lam)/(u-ell)
        eta=(ell+u)/(u-ell)
        res *= cheb_T(int(na),z)/cheb_T(int(na),np.array([eta]))[0]
    return res

def contraction(eigs,weights,blocks,n):
    q=block_residual(eigs,blocks,n)
    return float(np.sqrt(np.sum(weights*q*q)))

def contraction_single(eigs,weights,N,mu,L):
    q=cheb_residual_interval(eigs,N,mu,L)
    return float(np.sqrt(np.sum(weights*q*q)))

def quantiles(values,axis=0):
    return np.nanpercentile(values,[25,50,75],axis=axis)

# ---------- ensemble main ----------
def sample_clustered_instances(blocks,count,eigs_per_block,seed):
    rng=np.random.default_rng(seed)
    inst=[]
    total=eigs_per_block*len(blocks)
    for _ in range(count):
        eigs=[]
        for ell,u in blocks:
            # beta samples create different within-block clustering, still inside declared blocks.
            a=1.0+2.5*rng.random(); b=1.0+2.5*rng.random()
            vals=ell+(u-ell)*rng.beta(a,b,size=eigs_per_block)
            eigs.extend(vals)
        eigs=np.array(eigs,float)
        w=rng.gamma(shape=0.25,scale=1.0,size=total)
        w=w/np.sum(w)
        inst.append((eigs,w))
    return inst

def run_main():
    blocks=CONFIG['base_blocks']; Ns=CONFIG['N_grid']; Nmain=CONFIG['N_main']
    mu=min(b[0] for b in blocks); L=max(b[1] for b in blocks)
    lam_grid=spectral_grid(blocks,CONFIG['residual_grid_per_block'])
    inst=sample_clustered_instances(blocks,CONFIG['ensemble_instances'],CONFIG['eigs_per_block'],CONFIG['seed'])
    rows=[]; ens_rows=[]; env_rows=[]
    for N in Ns:
        n,theta=greedy_allocation(blocks,N)
        res_block=np.max(np.abs(block_residual(lam_grid,blocks,n)))
        res_single=np.max(np.abs(cheb_residual_interval(lam_grid,N,mu,L)))
        vals_block=[]; vals_single=[]; ratios=[]
        for j,(eigs,w) in enumerate(inst):
            cb=contraction(eigs,w,blocks,n)
            cs=contraction_single(eigs,w,N,mu,L)
            vals_block.append(cb); vals_single.append(cs); ratios.append(cs/max(cb,1e-300))
            ens_rows.append({'N':N,'instance':j,'block':cb,'single_block':cs,'improvement_ratio':cs/max(cb,1e-300)})
        q_b=quantiles(np.array(vals_block)); q_s=quantiles(np.array(vals_single)); q_r=quantiles(np.array(ratios))
        rows.append({'N':N,'Theta':theta,'cert':math.exp(theta),'sup_block':res_block,'sup_single_block':res_single,
                     'block_q25':q_b[0],'block_med':q_b[1],'block_q75':q_b[2],
                     'single_q25':q_s[0],'single_med':q_s[1],'single_q75':q_s[2],
                     'ratio_q25':q_r[0],'ratio_med':q_r[1],'ratio_q75':q_r[2],
                     **{f'n{a+1}':int(n[a]) for a in range(len(blocks))}})
    df=pd.DataFrame(rows); edf=pd.DataFrame(ens_rows)
    df.to_csv(TABLES/'main_summary_by_horizon.csv',index=False)
    edf.to_csv(TABLES/'main_ensemble_contractions.csv',index=False)
    # horizon to tolerance per instance
    tol=CONFIG['tol']
    hrows=[]
    for j in range(CONFIG['ensemble_instances']):
        sub=edf[edf.instance==j].sort_values('N')
        for method in ['block','single_block']:
            ok=sub[sub[method]<=tol]
            hit=int(ok.N.iloc[0]) if len(ok) else np.nan
            hrows.append({'instance':j,'method':method,'hit_N':hit})
    hdf=pd.DataFrame(hrows); hdf.to_csv(TABLES/'horizon_to_tolerance.csv',index=False)
    # residual envelope at Nmain
    nmain,_=greedy_allocation(blocks,Nmain)
    for la,rr in zip(lam_grid,block_residual(lam_grid,blocks,nmain)):
        env_rows.append({'method':'block','lambda':la,'abs_residual':abs(rr)})
    for la,rr in zip(lam_grid,cheb_residual_interval(lam_grid,Nmain,mu,L)):
        env_rows.append({'method':'single_block','lambda':la,'abs_residual':abs(rr)})
    pd.DataFrame(env_rows).to_csv(TABLES/'residual_envelope_N45.csv',index=False)
    return df,edf,hdf,lam_grid,nmain

def figure_main(df,edf,hdf,lam_grid,nmain):
    blocks=CONFIG['base_blocks']; Nmain=CONFIG['N_main']; mu=min(b[0] for b in blocks); L=max(b[1] for b in blocks)
    fig,axs=plt.subplots(2,2,figsize=(10.9,7.2),constrained_layout=True)
    ax=axs[0,0]
    for ell,u in blocks: ax.axvspan(ell,u,color=COL['gray'],alpha=.065,lw=0)
    for bi,(ell,u) in enumerate(blocks):
        xs=np.linspace(ell,u,CONFIG['residual_grid_per_block'])
        ax.plot(xs,np.abs(block_residual(xs,blocks,nmain)),color=COL['block'],label=(f'Block-reducible Jacobi {tuple(int(x) for x in nmain)}' if bi==0 else None))
        ax.plot(xs,np.abs(cheb_residual_interval(xs,Nmain,mu,L)),color=COL['single'],ls='--',label=('Single-block Jacobi' if bi==0 else None))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'spectral point $\lambda$')
    ax.set_ylabel(r'$|p_N(\lambda)|$')
    ax.set_title(f'Residual envelope at $N={Nmain}$'); legend(ax,'lower left'); panel(ax,'A')

    ax=axs[0,1]
    N=df.N.to_numpy(float)
    rng_sc=np.random.default_rng(CONFIG['seed']+101)
    for method,c in [('block',COL['block']),('single_block',COL['single'])]:
        sub=edf[['N',method]].rename(columns={method:'value'})
        xs=sub.N.to_numpy(float)+rng_sc.uniform(-0.22,0.22,size=len(sub))
        ax.scatter(xs,sub.value.to_numpy(float),s=3.0,alpha=.055,color=c,linewidths=0)
    ax.fill_between(N,df.block_q25,df.block_q75,color=COL['block'],alpha=.18,lw=0)
    ax.plot(N,df.block_med,color=COL['block'],label='Block-reducible Jacobi')
    ax.fill_between(N,df.single_q25,df.single_q75,color=COL['single'],alpha=.15,lw=0)
    ax.plot(N,df.single_med,color=COL['single'],ls='--',label='Single-block Jacobi')
    ax.set_yscale('log'); ax.set_xlabel('horizon $N$'); ax.set_ylabel(r'terminal contraction')
    ax.set_title('Clustered quadratic ensemble')
    legend(ax,'upper right'); panel(ax,'B')

    ax=axs[1,0]
    sub=edf[['N','improvement_ratio']]
    xs=sub.N.to_numpy(float)+rng_sc.uniform(-0.22,0.22,size=len(sub))
    ax.scatter(xs,sub.improvement_ratio.to_numpy(float),s=3.0,alpha=.055,color=COL['gold'],linewidths=0)
    ax.fill_between(N,df.ratio_q25,df.ratio_q75,color=COL['gold'],alpha=.24,lw=0)
    ax.plot(N,df.ratio_med,color=COL['gold'],label='median with IQR')
    ax.axhline(1,color=COL['gray'],lw=.9,ls=':')
    ax.set_yscale('log'); ax.set_xlabel('horizon $N$'); ax.set_ylabel(r'single/block contraction ratio')
    ax.set_title('Paired improvement over ensemble')
    legend(ax,'upper left'); panel(ax,'C')

    ax=axs[1,1]
    # paired strip with jitter, medians and IQRs, no histograms.
    rng=np.random.default_rng(CONFIG['seed']+11)
    methods=[('block','Block',COL['block']),('single_block','Single block',COL['single'])]
    for i,(m,label,c) in enumerate(methods):
        vals=hdf[hdf.method==m].hit_N.to_numpy(float)
        vals=vals[np.isfinite(vals)]
        x=i + rng.uniform(-0.09,0.09,size=vals.size)
        ax.scatter(x,vals,s=9,alpha=.28,color=c,linewidths=0)
        q=np.percentile(vals,[25,50,75])
        ax.vlines(i,q[0],q[2],color=c,lw=3.2)
        ax.plot([i-.15,i+.15],[q[1],q[1]],color=c,lw=2.2)
    ax.set_xticks([0,1],['Block','Single'])
    ax.set_ylabel(fr'first $N$ reaching ${CONFIG["tol"]:.0e}$')
    ax.set_title('Horizon-to-tolerance distribution')
    panel(ax,'D')
    save(fig,'spectrum_adapted_block_ensemble_main')

# ---------- leakage ----------
def actual_support_for_delta(base_blocks,delta):
    pts=[]
    for ell,u in base_blocks:
        width=u-ell
        lo=max(1e-8,ell-delta*width)
        hi=u+delta*width
        pts.extend(np.linspace(lo,hi,160))
    for (ell,u),(ell2,u2) in zip(base_blocks[:-1],base_blocks[1:]):
        gap=ell2-u
        if delta>0:
            pts.extend(u+np.linspace(0,delta*gap,70))
            pts.extend(ell2-np.linspace(0,delta*gap,70))
    return np.array(sorted(set(np.round(pts,12))),float)

def sample_leaked_instances(base_blocks,delta,count,eigs_per_block,seed):
    rng=np.random.default_rng(seed+int(round(delta*10000)))
    inst=[]
    for _ in range(count):
        eigs=[]
        # Most eigenvalues remain in expanded collars near blocks; some are in gap collars.
        for ell,u in base_blocks:
            width=u-ell
            lo=max(1e-8,ell-delta*width)
            hi=u+delta*width
            eigs.extend(rng.uniform(lo,hi,size=eigs_per_block))
        if delta>0:
            gap_count=max(1,eigs_per_block//5)
            for (ell,u),(ell2,u2) in zip(base_blocks[:-1],base_blocks[1:]):
                gap=ell2-u
                eigs.extend(u+rng.uniform(0,delta*gap,size=gap_count))
                eigs.extend(ell2-rng.uniform(0,delta*gap,size=gap_count))
        eigs=np.array(eigs,float)
        w=rng.gamma(0.35,1.0,size=eigs.size); w=w/w.sum()
        inst.append((eigs,w))
    return inst

def run_leakage():
    base=CONFIG['base_blocks']; guard=CONFIG['guarded_blocks']; ghull=CONFIG['guarded_hull']
    N=CONFIG['N_main']; mu=min(b[0] for b in base); L=max(b[1] for b in base)
    designs=[('nominal_block',base,COL['block']),('guarded_wide',guard,COL['guard']),('guarded_hull',ghull,COL['merge']),('single_block',[(mu,L)],COL['single'])]
    # Precompute allocations for designs.
    alloc={name:greedy_allocation(blocks,N)[0] for name,blocks,_ in designs}
    rows=[]; ens=[]
    for delta in CONFIG['leakage_deltas']:
        pts=actual_support_for_delta(base,float(delta))
        instances=sample_leaked_instances(base,float(delta),CONFIG['leakage_instances'],CONFIG['eigs_per_block'],CONFIG['seed']+500)
        for name,blocks,_ in designs:
            n=alloc[name]
            if name=='single_block':
                worst=float(np.max(np.abs(cheb_residual_interval(pts,N,mu,L))))
                vals=[contraction_single(e,w,N,mu,L) for e,w in instances]
            else:
                worst=float(np.max(np.abs(block_residual(pts,blocks,n))))
                vals=[contraction(e,w,blocks,n) for e,w in instances]
            q=np.percentile(vals,[25,50,75])
            rows.append({'delta':delta,'method':name,'worst_residual':worst,'q25':q[0],'median':q[1],'q75':q[2]})
            for j,v in enumerate(vals): ens.append({'delta':delta,'method':name,'instance':j,'contraction':v})
    df=pd.DataFrame(rows); edf=pd.DataFrame(ens)
    df.to_csv(TABLES/'leakage_summary.csv',index=False)
    edf.to_csv(TABLES/'leakage_ensemble.csv',index=False)
    return df,alloc

def figure_leakage(df,alloc):
    fig,axs=plt.subplots(1,2,figsize=(10.9,3.6),constrained_layout=True)
    labels={'nominal_block':'Nominal blocks','guarded_wide':'Guarded widened blocks','guarded_hull':'Guarded hull Jacobi','single_block':'Single-block Jacobi'}
    colors={'nominal_block':COL['block'],'guarded_wide':COL['guard'],'guarded_hull':COL['merge'],'single_block':COL['single']}
    lss={'nominal_block':'-','guarded_wide':'-.','guarded_hull':':','single_block':'--'}
    ax=axs[0]
    for m in ['nominal_block','guarded_wide','guarded_hull','single_block']:
        sub=df[df.method==m].sort_values('delta')
        ax.plot(100*sub.delta,sub.worst_residual,color=colors[m],ls=lss[m],label=labels[m])
    ax.set_yscale('log'); ax.set_xlabel('gap/collar leakage (%)'); ax.set_ylabel('worst residual on actual support')
    ax.set_title('Deterministic support leakage diagnostic')
    legend(ax,'upper left'); panel(ax,'A')
    ax=axs[1]
    for m in ['nominal_block','guarded_wide','guarded_hull','single_block']:
        sub=df[df.method==m].sort_values('delta')
        x=100*sub.delta.to_numpy(float)
        ax.fill_between(x,sub.q25,sub.q75,color=colors[m],alpha=.13,lw=0)
        ax.plot(x,sub['median'],color=colors[m],ls=lss[m],label=labels[m])
    ax.set_yscale('log'); ax.set_xlabel('gap/collar leakage (%)'); ax.set_ylabel('ensemble terminal contraction')
    ax.set_title('Random leaked clustered quadratics')
    legend(ax,'upper left'); panel(ax,'B')
    save(fig,'spectrum_adapted_leakage_diagnostic')



# ---------- v5 visual overrides ----------
def _local_block_axis(blocks, per_block):
    xs_all=[]; lam_all=[]; centers=[]; labels=[]
    gap=0.12
    for i,(ell,u) in enumerate(blocks):
        t=np.linspace(0,1,per_block)
        x=i*(1+gap)+t
        lam=ell+(u-ell)*t
        xs_all.append(x); lam_all.append(lam)
        centers.append(i*(1+gap)+0.5)
        labels.append(fr'$E_{i+1}$\n[{ell:g},{u:g}]')
    return xs_all,lam_all,centers,labels,gap

def figure_main(df,edf,hdf,lam_grid,nmain):
    blocks=CONFIG['base_blocks']; Nmain=CONFIG['N_main']; mu=min(b[0] for b in blocks); L=max(b[1] for b in blocks)
    fig,axs=plt.subplots(2,2,figsize=(10.9,7.2),constrained_layout=True)
    ax=axs[0,0]
    xs_all,lam_all,centers,labels,gap=_local_block_axis(blocks,760)
    for i,(xs,lam) in enumerate(zip(xs_all,lam_all)):
        ax.axvspan(xs[0],xs[-1],color=COL['gray'],alpha=.055,lw=0)
        rb=np.abs(block_residual(lam,blocks,nmain))
        rs=np.abs(cheb_residual_interval(lam,Nmain,mu,L))
        ax.plot(xs,rb,color=COL['block'],lw=1.75,label=(f'Block-reducible Jacobi {tuple(int(x) for x in nmain)}' if i==0 else None))
        ax.plot(xs,rs,color=COL['single'],lw=1.55,ls='--',label=('Single-block Jacobi on hull' if i==0 else None))
        # show sampled envelope points so the panel is visually dense and tied to the support grid.
        idx=np.linspace(0,len(xs)-1,42,dtype=int)
        ax.scatter(xs[idx],rb[idx],s=6,color=COL['block'],alpha=.35,linewidths=0)
        ax.scatter(xs[idx],rs[idx],s=5,color=COL['single'],alpha=.28,linewidths=0)
    for i in range(1,len(blocks)):
        ax.axvline(i*(1+gap)-gap/2,color=COL['gray'],lw=.8,ls=':',alpha=.6)
    ax.set_yscale('log')
    ax.set_xlim(xs_all[0][0],xs_all[-1][-1])
    ax.set_xticks(centers,labels)
    ax.set_xlabel('declared spectral blocks')
    ax.set_ylabel(r'$|p_N(\lambda)|$ on support')
    ax.set_title(f'Residual envelope on declared support at $N={Nmain}$')
    legend(ax,'lower left'); panel(ax,'A')

    ax=axs[0,1]
    N=df.N.to_numpy(float)
    rng_sc=np.random.default_rng(CONFIG['seed']+101)
    for method,c in [('block',COL['block']),('single_block',COL['single'])]:
        sub=edf[['N',method]].rename(columns={method:'value'})
        xs=sub.N.to_numpy(float)+rng_sc.uniform(-0.22,0.22,size=len(sub))
        ax.scatter(xs,sub.value.to_numpy(float),s=3.0,alpha=.055,color=c,linewidths=0)
    ax.fill_between(N,df.block_q25,df.block_q75,color=COL['block'],alpha=.18,lw=0)
    ax.plot(N,df.block_med,color=COL['block'],label='Block-reducible Jacobi')
    ax.fill_between(N,df.single_q25,df.single_q75,color=COL['single'],alpha=.15,lw=0)
    ax.plot(N,df.single_med,color=COL['single'],ls='--',label='Single-block Jacobi')
    ax.set_yscale('log'); ax.set_xlabel('horizon $N$'); ax.set_ylabel(r'terminal contraction')
    ax.set_title('Clustered quadratic ensemble')
    legend(ax,'upper right'); panel(ax,'B')

    ax=axs[1,0]
    sub=edf[['N','improvement_ratio']]
    xs=sub.N.to_numpy(float)+rng_sc.uniform(-0.22,0.22,size=len(sub))
    ax.scatter(xs,sub.improvement_ratio.to_numpy(float),s=3.0,alpha=.055,color=COL['gold'],linewidths=0)
    ax.fill_between(N,df.ratio_q25,df.ratio_q75,color=COL['gold'],alpha=.24,lw=0)
    ax.plot(N,df.ratio_med,color=COL['gold'],label='median with IQR')
    ax.axhline(1,color=COL['gray'],lw=.9,ls=':')
    ax.set_yscale('log'); ax.set_xlabel('horizon $N$'); ax.set_ylabel(r'single/block contraction ratio')
    ax.set_title('Paired improvement over ensemble')
    legend(ax,'upper left'); panel(ax,'C')

    ax=axs[1,1]
    rng=np.random.default_rng(CONFIG['seed']+11)
    methods=[('block','Block',COL['block']),('single_block','Single block',COL['single'])]
    for i,(m,label,c) in enumerate(methods):
        vals=hdf[hdf.method==m].hit_N.to_numpy(float)
        vals=vals[np.isfinite(vals)]
        x=i + rng.uniform(-0.09,0.09,size=vals.size)
        ax.scatter(x,vals,s=9,alpha=.28,color=c,linewidths=0)
        q=np.percentile(vals,[25,50,75])
        ax.vlines(i,q[0],q[2],color=c,lw=3.2)
        ax.plot([i-.15,i+.15],[q[1],q[1]],color=c,lw=2.2)
    ax.set_xticks([0,1],['Block','Single'])
    ax.set_ylabel(fr'first $N$ reaching ${CONFIG["tol"]:.0e}$')
    ax.set_title('Horizon-to-tolerance distribution')
    panel(ax,'D')
    save(fig,'spectrum_adapted_block_ensemble_main')

def figure_leakage(df,alloc):
    fig,axs=plt.subplots(1,2,figsize=(10.9,3.65),constrained_layout=True)
    labels={'nominal_block':'Nominal narrow blocks','guarded_wide':'Guarded widened blocks','guarded_hull':'Guarded hull Jacobi','single_block':'Single-block Jacobi'}
    colors={'nominal_block':COL['block'],'guarded_wide':COL['guard'],'guarded_hull':COL['merge'],'single_block':COL['single']}
    lss={'nominal_block':'-','guarded_wide':'-.','guarded_hull':':','single_block':'--'}
    ax=axs[0]
    for m in ['nominal_block','guarded_wide','guarded_hull','single_block']:
        sub=df[df.method==m].sort_values('delta')
        ax.plot(100*sub.delta,sub.worst_residual,color=colors[m],ls=lss[m],label=labels[m],lw=1.75)
    ax.set_yscale('log'); ax.set_xlabel('support leakage level (%)'); ax.set_ylabel('worst residual on actual support')
    ax.set_title('Deterministic support leakage diagnostic')
    legend(ax,'upper left'); panel(ax,'A')
    ax=axs[1]
    edf=pd.read_csv(TABLES/'leakage_ensemble.csv')
    rng=np.random.default_rng(CONFIG['seed']+808)
    for m in ['nominal_block','guarded_wide','guarded_hull','single_block']:
        sub=df[df.method==m].sort_values('delta')
        x=100*sub.delta.to_numpy(float)
        raw=edf[edf.method==m]
        # low-alpha points give the empirical spread without using histograms.
        ax.scatter(100*raw.delta.to_numpy(float)+rng.uniform(-0.055,0.055,size=len(raw)),
                   raw.contraction.to_numpy(float),s=2.2,alpha=.035,color=colors[m],linewidths=0)
        ax.fill_between(x,sub.q25,sub.q75,color=colors[m],alpha=.14,lw=0)
        ax.plot(x,sub['median'],color=colors[m],ls=lss[m],label=labels[m],lw=1.7)
    ax.set_yscale('log'); ax.set_xlabel('support leakage level (%)'); ax.set_ylabel('ensemble terminal contraction')
    ax.set_title('Random leaked clustered quadratics')
    legend(ax,'upper left'); panel(ax,'B')
    save(fig,'spectrum_adapted_leakage_diagnostic')

if __name__=='__main__':
    df_main,edf,hdf,lam_grid,nmain=run_main()
    figure_main(df_main,edf,hdf,lam_grid,nmain)
    df_leak,alloc=run_leakage()
    figure_leakage(df_leak,alloc)
    print(f'Wrote {OUT} and {ROOT/"experiment_spectrum_adapted_v6.zip"}')
