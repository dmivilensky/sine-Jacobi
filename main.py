#!/usr/bin/env python3
from __future__ import annotations
import math, json, shutil, zipfile, sys, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path('/mnt/data')
OUT=ROOT/'experiment_v18_targeted_final'
FIGS=OUT/'figures'; TABLES=OUT/'tables'; CACHE=ROOT/'experiment_v18_targeted_final_cache'
if OUT.exists() and os.environ.get('V18_RESUME')!='1': shutil.rmtree(OUT)
FIGS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True); CACHE.mkdir(exist_ok=True)

sys.path.insert(0,str(ROOT))
from exact_realizations import PrefixChebyshev, SineJacobi
from cheb_robustness import make_endpoint_coupled_logistic, endpoint_directions, cheb_epsilon

CONFIG={
  'finite_constants': {'mu':1.0,'L':120.0,'N_min':8,'N_max':280,'overhead_tau':0.25},
  'stochastic_frontier': {
    'mu':1.0,'L':120.0,'kappa':120.0,'active_modes':[1.0,'sqrt(kappa)','kappa'],
    'initial_endpoint_mid':[1.0,0.25,1.0],
    'noise_sigma':2.0,
    'trials':900,
    'N_grid':list(range(16,101,2)),
    'batch_grid':[64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,896,1024,1152,1280,1408,1536,1792,2048],
    'overhead_tau':0.25,
    'seed':202706,
  },
  'nonlinear_family': {
    'mu':1.0,'L':1e4,'N':256,'rho_grid_points':121,
    'instances': [
      {'center':0.55,'width':0.13,'q':2.20}, {'center':0.62,'width':0.15,'q':2.45},
      {'center':0.70,'width':0.17,'q':2.65}, {'center':0.78,'width':0.14,'q':2.85},
      {'center':0.66,'width':0.11,'q':2.55},
    ],
  }
}

COL={'prefix':'#315f9b','sine':'#24804c','purple':'#7254a2','orange':'#c66a2d','red':'#a94442','gray':'#6a6a6a','black':'#202020','blue2':'#6d9ac9','green2':'#73a66b','gold':'#c99a2e'}
LAB={'prefix':'Prefix-Chebyshev','sine':'Sine-Jacobi'}
MARK={'prefix':'o','sine':'s'}
plt.rcParams.update({
    'figure.dpi':190,'savefig.dpi':430,
    'font.size':9.4,'axes.labelsize':9.6,'axes.titlesize':10.2,'legend.fontsize':7.4,
    'xtick.labelsize':8.2,'ytick.labelsize':8.2,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.55,
    'mathtext.fontset':'dejavusans','lines.linewidth':1.6,
})

def save(fig,name):
    fig.savefig(FIGS/f'{name}.pdf',bbox_inches='tight')
    # PNGs are only for contact sheets/previews; PDFs remain the article-quality output.
    fig.savefig(FIGS/f'{name}.png',dpi=170)
    plt.close(fig)

def panel(ax,l):
    ax.text(-0.105,1.06,l,transform=ax.transAxes,ha='left',va='top',fontsize=11,fontweight='bold')

def legend(ax,loc='best',**kw):
    ax.legend(loc=loc,frameon=True,framealpha=.96,edgecolor='none',handlelength=2.1,columnspacing=.8,**kw)

def moving_average(y,window=7):
    y=np.asarray(y,float)
    if len(y)<window: return y
    if window%2==0: window+=1
    pad=window//2
    return np.convolve(np.pad(y,(pad,pad),mode='edge'),np.ones(window)/window,mode='valid')

def stable_cheb_eps(N,mu=1.0,L=120.0):
    eta=(L+mu)/(L-mu)
    return 1.0/np.cosh(np.asarray(N)*math.acosh(eta))

# exact constants -------------------------------------------------------------
def sine_a_array(N:int):
    if N<=1: return np.zeros(0)
    s=np.arange(1,N)
    a2=0.25*(np.sin(np.pi*s/N)**2)/(np.sin(np.pi*(s-.5)/N)*np.sin(np.pi*(s+.5)/N))
    return np.sqrt(a2)

def tridiag_inverse_diag_I_minus_J_sine(N:int):
    if N<=0: return np.zeros(0)
    a=sine_a_array(N)
    theta=np.zeros(N+1); theta[0]=1.0; theta[1]=1.0
    for i in range(2,N+1): theta[i]=theta[i-1]-(a[i-2]**2)*theta[i-2]
    phi=np.zeros(N+2); phi[N+1]=1.0; phi[N]=1.0
    for i in range(N-1,0,-1): phi[i]=phi[i+1]-(a[i-1]**2)*phi[i+2]
    return np.array([theta[i]*phi[i+2]/theta[N] for i in range(N)])

def finite_constants():
    cfg=CONFIG['finite_constants']; mu=cfg['mu']; L=cfg['L']; Delta=L-mu
    rows=[]
    for N in range(cfg['N_min'],cfg['N_max']+1):
        eps=float(stable_cheb_eps(N,mu,L))
        Cpref=float(2.0*math.sqrt(N*N+(2.0/3.0)*N*(N-1)*(2*N-1))/(N**1.5))
        d=tridiag_inverse_diag_I_minus_J_sine(N)
        Csine=float(2.0*np.sqrt(np.sum(d*d))/(N**1.5))
        ap=(Cpref*(N**1.5)/Delta)**2
        asi=(Csine*(N**1.5)/Delta)**2
        rows.append({'N':N,'eps':eps,'C_pref':Cpref,'C_sine':Csine,'squared_ratio':(Csine/Cpref)**2,
                     'a_prefix':ap,'a_sine':asi,'bmin_prefix_tau':ap/cfg['overhead_tau'],'bmin_sine_tau':asi/cfg['overhead_tau']})
    df=pd.DataFrame(rows); df.to_csv(TABLES/'finite_constants.csv',index=False); return df

def figure_theory_constants(df):
    fig,axs=plt.subplots(1,3,figsize=(12.3,3.45),constrained_layout=True)
    ax=axs[0]
    ax.plot(df.N,df.C_pref,color=COL['prefix'],label='Prefix finite')
    ax.plot(df.N,df.C_sine,color=COL['sine'],label='Sine finite')
    ax.axhline(4/math.sqrt(3),color=COL['prefix'],ls='--',lw=1,label='Prefix limit')
    ax.axhline(2*math.sqrt(1.1426922),color=COL['sine'],ls='--',lw=1,label='Sine limit')
    ax.set_xscale('log'); ax.set_xlabel('horizon $N$'); ax.set_ylabel(r'$C_N$'); ax.set_title('Finite first-order constants'); legend(ax,'lower right'); panel(ax,'A')
    ax=axs[1]
    ax.plot(df.N,df.squared_ratio,color=COL['purple'],label='finite squared ratio')
    ax.axhline((2*math.sqrt(1.1426922)/(4/math.sqrt(3)))**2,color=COL['black'],ls='--',lw=1,label='asymptotic ratio')
    ax.set_xscale('log'); ax.set_ylim(.84,.935); ax.set_xlabel('horizon $N$'); ax.set_ylabel(r'$(C_N^{\rm sine}/C_N^{\rm pref})^2$'); ax.set_title('Relative drift coefficient'); legend(ax,'upper right'); panel(ax,'B')
    ax=axs[2]
    savepct=100*(1-df.bmin_sine_tau/df.bmin_prefix_tau)
    ax.plot(df.N,savepct,color=COL['orange'],label='equivalent batch saving')
    ax.fill_between(df.N,np.minimum.accumulate(savepct[::-1])[::-1],np.maximum.accumulate(savepct),color=COL['orange'],alpha=.08)
    ax.set_xscale('log'); ax.set_xlabel('horizon $N$'); ax.set_ylabel('saving at same overhead (%)'); ax.set_title('Coefficient translated to batch saving'); legend(ax,'lower right'); panel(ax,'C')
    save(fig,'v18t_theory_constants')

# stochastic empirical frontier ----------------------------------------------
def run_stoch_method(method,N,b,z,eigs,x0,sigma):
    trials=z.shape[0]
    Xprev=np.tile(x0,(trials,1))
    def grad(X,t):
        hx=X*eigs[None,:]
        gx=np.zeros_like(X)
        gx[:,0]=X[:,-1]
        gx[:,-1]=X[:,0]
        return hx + (sigma/math.sqrt(b))*z[:,t][:,None]*gx
    if method=='prefix':
        meth=PrefixChebyshev(float(eigs[0]),float(eigs[-1]),int(N)); betas=meth.beta()
        Xcur=Xprev-(2.0/(eigs[-1]+eigs[0]))*grad(Xprev,0)
        for t in range(1,N):
            bb=betas[t-1]
            Xnext=Xcur+bb*(Xcur-Xprev)-(1+bb)*(2.0/(eigs[-1]+eigs[0]))*grad(Xcur,t)
            Xprev,Xcur=Xcur,Xnext
        return Xcur
    else:
        meth=SineJacobi(float(eigs[0]),float(eigs[-1]),int(N)); alpha,beta,gamma=meth.coefficients()
        Xcur=Xprev-(2.0/(eigs[-1]+eigs[0]))*grad(Xprev,0)
        for s in range(1,N):
            Xnext=alpha[s-1]*Xcur-beta[s-1]*Xprev-gamma[s-1]*grad(Xcur,s)
            Xprev,Xcur=Xcur,Xnext
        return Xcur

def exact_fixed_sq(method,N,eigs,x0):
    meth=PrefixChebyshev(float(eigs[0]),float(eigs[-1]),int(N)) if method=='prefix' else SineJacobi(float(eigs[0]),float(eigs[-1]),int(N))
    return float(np.sum(meth.run_gradient(lambda x:eigs*x,x0)**2))

def empirical_stochastic_frontier():
    cache=CACHE/'stochastic_frontier_v18.csv'
    if cache.exists():
        return pd.read_csv(cache)
    cfg=CONFIG['stochastic_frontier']; mu=cfg['mu']; L=cfg['L']; sigma=cfg['noise_sigma']; trials=cfg['trials']
    eigs=np.array([mu,math.sqrt(L),L],float)
    x0=np.array(cfg['initial_endpoint_mid'],float); x0=x0/np.linalg.norm(x0)
    maxN=max(cfg['N_grid']); rng=np.random.default_rng(cfg['seed'])
    rows=[]
    for b in cfg['batch_grid']:
        # Common random numbers for all N and both methods at a fixed batch.
        zbase=rng.normal(size=(trials,maxN+1))
        for N in cfg['N_grid']:
            z=zbase[:,:N]
            for method in ['prefix','sine']:
                X=run_stoch_method(method,N,b,z,eigs,x0,sigma)
                sq=np.sum(X*X,axis=1)
                fixed=exact_fixed_sq(method,N,eigs,x0)
                vals=sq/max(fixed,1e-300)-1.0
                rows.append({'method':method,'N':int(N),'batch':int(b),'sigma':float(sigma),'trials':int(trials),
                             'fixed_sq':fixed,'mean_over_fixed':float(np.mean(sq/fixed)),
                             'overhead':float(np.mean(vals)),'se_overhead':float(np.std(vals,ddof=1)/math.sqrt(trials)),
                             'q25_overhead':float(np.quantile(vals,.25)),'q75_overhead':float(np.quantile(vals,.75)),
                             'mean_sq':float(np.mean(sq))})
    df=pd.DataFrame(rows)
    df.to_csv(cache,index=False); return df

def select_frontier(df,tau):
    rows=[]
    for b,g0 in df.groupby('batch'):
        for method,g in g0.groupby('method'):
            g=g.sort_values('N')
            ok=g[g.overhead<=tau]
            if len(ok)==0:
                rows.append({'batch':b,'method':method,'N_safe':np.nan,'logdecay_per_grad':np.nan,'grad_per_log':np.nan,'overhead_at_safe':np.nan})
                continue
            # Choose the admissible N with largest empirical per-gradient log decrease.
            gg=ok.copy()
            gg['logdecay_per_grad']=-np.log(np.maximum(gg.mean_sq.to_numpy(float),1e-300))/gg.N.to_numpy(float)
            r=gg.loc[gg.logdecay_per_grad.idxmax()]
            rows.append({'batch':b,'method':method,'N_safe':float(r.N),'logdecay_per_grad':float(r.logdecay_per_grad),
                         'grad_per_log':float(1/max(r.logdecay_per_grad,1e-300)),'overhead_at_safe':float(r.overhead)})
    return pd.DataFrame(rows)

def figure_optimization(emp):
    tau=CONFIG['stochastic_frontier']['overhead_tau']
    front=select_frontier(emp,tau)
    front.to_csv(TABLES/'empirical_stochastic_frontier.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(10.8,7.05),constrained_layout=True)
    ax=axs[0,0]
    show_batches=[96,192,384,768,1536]
    cmap=plt.get_cmap('viridis')
    norm=plt.Normalize(min(show_batches),max(show_batches))
    for b in show_batches:
        color=cmap(norm(b))
        for method in ['prefix','sine']:
            d=emp[(emp.batch==b)&(emp.method==method)].sort_values('N')
            if d.empty: continue
            ls='-' if method=='prefix' else '--'
            y=d.overhead.to_numpy(float); se=d.se_overhead.to_numpy(float); x=d.N.to_numpy(float)
            ax.fill_between(x, np.maximum(y-1.96*se, 1e-12), y+1.96*se, color=color, alpha=.085, linewidth=0)
            ax.plot(x, y, color=color, alpha=.82, ls=ls, lw=1.05)
    ax.axhline(tau,color=COL['black'],ls=':',lw=1.15)
    ax.set_yscale('log'); ax.set_xlabel('block horizon $N$'); ax.set_ylabel('empirical overhead over fixed-Hessian block'); ax.set_title('Stress-test overhead curves')
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.82); cb.set_label('batch/noise budget $b$')
    ax.legend(handles=[Line2D([0],[0],color='0.25',marker='o',lw=1,label='Prefix'),Line2D([0],[0],color='0.25',marker='s',lw=1,ls='--',label='Sine'),Line2D([0],[0],color=COL['black'],ls=':',label=f'{int(100*tau)}% rule')],loc='upper left',frameon=True,framealpha=.96,edgecolor='none'); panel(ax,'A')
    ax=axs[0,1]
    for method in ['prefix','sine']:
        d=front[front.method==method].sort_values('batch')
        ax.plot(d.batch,d.N_safe,color=COL[method],marker=MARK[method],ms=3.4,label=LAB[method])
    ax.set_xscale('log'); ax.set_xlabel('batch/noise budget $b$'); ax.set_ylabel('selected admissible horizon'); ax.set_title('Empirical robust block frontier'); legend(ax,'upper left'); panel(ax,'B')
    ax=axs[1,0]
    fp=front[front.method=='prefix'].set_index('batch'); fs=front[front.method=='sine'].set_index('batch'); common=sorted(set(fp.index)&set(fs.index))
    delta=np.array([fs.loc[b,'N_safe']-fp.loc[b,'N_safe'] for b in common],float)
    ax.axhline(0,color=COL['black'],lw=1,ls='--')
    ax.plot(common, delta, color=COL['purple'], lw=1.6, label='extra admissible horizon')
    ax.set_xscale('log'); ax.set_xlabel('batch/noise budget $b$'); ax.set_ylabel(r'$N_{\rm sine}-N_{\rm pref}$'); ax.set_title('Extra safe horizon'); ax.set_ylim(-.4,max(5,float(np.nanmax(delta)+1))); legend(ax,'upper left'); panel(ax,'C')
    ax=axs[1,1]
    saving=np.array([100*(1-fs.loc[b,'grad_per_log']/fp.loc[b,'grad_per_log']) for b in common],float)
    ax.axhline(0,color=COL['black'],lw=1,ls='--')
    ax.fill_between(common,0,saving,color=COL['orange'],alpha=.20)
    ax.plot(common,saving,color=COL['orange'],lw=1.65,label='oracle-efficiency saving')
    ax.set_xscale('log'); ax.set_xlabel('batch/noise budget $b$'); ax.set_ylabel('saving (%)'); ax.set_title('Empirical oracle-efficiency gain'); legend(ax,'upper left'); panel(ax,'D')
    save(fig,'v18t_optimization_frontier')

# TVQ packet validation -------------------------------------------------------
def load_packet_data():
    src=ROOT/'cheb_sine_article_experiments_v2_audited'/'outputs'/'tables'
    packet=pd.read_csv(src/'B_localized_random_packet_rms.csv')
    transfer=pd.read_csv(src/'F8_robustness_transfer_law.csv')
    return packet,transfer

def figure_tvq(packet,transfer):
    fig,axs=plt.subplots(2,2,figsize=(9.55,6.22),constrained_layout=True)
    ax=axs[0,0]
    for method,pcol,mcol in [('prefix','prefix_predicted_rms','prefix_mc_rms'),('sine','sine_predicted_rms','sine_mc_rms')]:
        ax.scatter(packet[pcol],packet[mcol],s=10,facecolors='none',edgecolors=COL[method],linewidth=.6,label=LAB[method])
    lo=min(packet.prefix_predicted_rms.min(),packet.sine_predicted_rms.min(),packet.prefix_mc_rms.min(),packet.sine_mc_rms.min()); hi=max(packet.prefix_predicted_rms.max(),packet.sine_predicted_rms.max(),packet.prefix_mc_rms.max(),packet.sine_mc_rms.max())
    ax.plot([lo,hi],[lo,hi],color=COL['black'],ls='--',lw=1,label='identity')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('kernel-predicted RMS'); ax.set_ylabel('Monte Carlo RMS'); ax.set_title('Exact-kernel calibration'); legend(ax,'upper left'); panel(ax,'A')
    ax=axs[0,1]
    vals=[packet.prefix_mc_over_predicted,packet.sine_mc_over_predicted,transfer.actual_ratio/transfer.predicted_ratio]
    labels=['Prefix RMS','Sine RMS','Ratio transfer']; colors=[COL['prefix'],COL['sine'],COL['purple']]
    bp=ax.boxplot(vals,tick_labels=labels,showfliers=False,patch_artist=True,widths=.55)
    for box,c in zip(bp['boxes'],colors): box.set(facecolor=c,alpha=.20,edgecolor=c,linewidth=1.1)
    for med in bp['medians']: med.set(color=COL['black'],lw=1.25)
    ax.axhline(1,color=COL['black'],ls='--',lw=1,label='perfect prediction')
    ax.set_ylabel('actual / predicted'); ax.set_title('Calibration ratios over packet grid')
    ax.legend(handles=[Patch(facecolor=COL['prefix'],edgecolor=COL['prefix'],alpha=.20,label='Prefix RMS'),
                       Patch(facecolor=COL['sine'],edgecolor=COL['sine'],alpha=.20,label='Sine RMS'),
                       Patch(facecolor=COL['purple'],edgecolor=COL['purple'],alpha=.20,label='ratio transfer'),
                       Line2D([0],[0],color=COL['black'],ls='--',label='perfect prediction')],
              loc='upper right',frameon=True,framealpha=.96,edgecolor='none'); panel(ax,'B')
    ax=axs[1,0]
    ax.scatter(packet.prefix_mc_rms,packet.sine_mc_rms,s=10,facecolors='none',edgecolors=COL['purple'],linewidth=.55,label='localized packets')
    lo=min(packet.prefix_mc_rms.min(),packet.sine_mc_rms.min()); hi=max(packet.prefix_mc_rms.max(),packet.sine_mc_rms.max()); med=float(np.median(packet.sine_prefix_mc_rms_ratio))
    ax.plot([lo,hi],[lo,hi],color=COL['black'],ls='--',lw=1,label='equal response')
    ax.plot([lo,hi],[med*lo,med*hi],color=COL['purple'],ls=':',lw=1.15,label=f'median ratio {med:.3f}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Prefix RMS response'); ax.set_ylabel('Sine RMS response'); ax.set_title('Paired packet response'); legend(ax,'upper left'); panel(ax,'C')
    ax=axs[1,1]
    # This is a grouped distribution, not a histogram: each box summarizes all packet locations at a fixed width.
    widths=np.array(sorted(packet.width.unique()))
    groups=[packet.loc[np.isclose(packet.width,w),'sine_prefix_mc_rms_ratio'].to_numpy() for w in widths]
    positions=np.arange(len(widths))
    bp=ax.boxplot(groups,positions=positions,showfliers=False,patch_artist=True,widths=.62)
    for box in bp['boxes']: box.set(facecolor=COL['purple'],alpha=.18,edgecolor=COL['purple'],linewidth=1.0)
    for medline in bp['medians']: medline.set(color=COL['black'],lw=1.15)
    ax.axhline(1,color=COL['black'],ls='--',lw=1,label='equal response')
    ticks=np.linspace(0,len(widths)-1,5,dtype=int); ax.set_xticks(ticks); ax.set_xticklabels([f'{widths[i]:.3f}' for i in ticks])
    ax.set_xlabel('packet width'); ax.set_ylabel('sine / prefix RMS'); ax.set_title('Width-conditioned response reduction'); legend(ax,'upper right'); panel(ax,'D')
    packet.to_csv(TABLES/'tvq_packet_grid.csv',index=False); transfer.to_csv(TABLES/'tvq_transfer_law.csv',index=False)
    save(fig,'v18t_tvq_kernel_validation')

# stochastic raw and collapse -------------------------------------------------
def figure_stochastic_and_collapse(emp):
    # Main stochastic figure: fixed horizon from the empirical grid, with design-tail coefficient summary.
    N0=44 if 44 in emp.N.unique() else int(np.median(emp.N.unique()))
    d0=emp[emp.N==N0].copy().sort_values('batch')
    tail_batches=sorted(emp.batch.unique())[-5:]
    rows=[]; boot=[]; rng=np.random.default_rng(991)
    for method in ['prefix','sine']:
        d=d0[(d0.method==method)&(d0.batch.isin(tail_batches))].sort_values('batch')
        x=1/d.batch.to_numpy(float); y=d.overhead.to_numpy(float)
        slope=float(np.sum(x*y)/np.sum(x*x)); se=float(np.sqrt(np.sum((y-slope*x)**2)/max(1,len(x)-1)/np.sum(x*x)))
        rows.append({'method':method,'N':N0,'design_tail_slope':slope,'se':se,'lo95':slope-1.96*se,'hi95':slope+1.96*se,'fit_batches':','.join(map(str,tail_batches))})
    fit=pd.DataFrame(rows); fit.to_csv(TABLES/'v18t_design_tail_coefficients.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(12.0,3.55),constrained_layout=True)
    ax=axs[0]
    for method in ['prefix','sine']:
        d=d0[d0.method==method]
        ax.errorbar(1/d.batch,d.overhead,yerr=1.96*d.se_overhead,fmt=MARK[method],ms=3.0,color=COL[method],ecolor=COL[method],capsize=1.8,linestyle='none',label=LAB[method])
        r=fit[fit.method==method].iloc[0]; xs=np.linspace((1/d.batch).min()*.85,(1/d.batch).max()*1.05,120)
        ax.plot(xs,r.design_tail_slope*xs,color=COL[method],ls='--',lw=1.1,label=f'{LAB[method]} tail fit')
    ax.set_xlabel(r'$1/b$'); ax.set_ylabel('empirical stochastic overhead'); ax.set_title(f'Batch scaling at $N={N0}$'); legend(ax,'upper left'); panel(ax,'A')
    ax=axs[1]
    xpos={'prefix':-0.055,'sine':0.055}
    for method in ['prefix','sine']:
        r=fit[fit.method==method].iloc[0]
        ax.errorbar([xpos[method]],[r.design_tail_slope],yerr=[[r.design_tail_slope-r.lo95],[r.hi95-r.design_tail_slope]],fmt=MARK[method],ms=5,color=COL[method],capsize=3,label=LAB[method])
    ratio=float(fit[fit.method=='sine'].design_tail_slope.iloc[0]/fit[fit.method=='prefix'].design_tail_slope.iloc[0])
    ax.text(.5,.90,f'sine/prefix slope = {ratio:.3f}',transform=ax.transAxes,ha='center',va='center')
    ax.set_xlim(-.32,.32); ax.set_xticks([xpos['prefix'],xpos['sine']]); ax.set_xticklabels(['Prefix','Sine']); ax.set_ylabel(r'tail coefficient $a_N$'); ax.set_title('Design-tail coefficient'); legend(ax,'upper right'); panel(ax,'B')
    ax=axs[2]
    by=[]
    for b,g in d0.groupby('batch'):
        p=g[g.method=='prefix'].overhead.iloc[0]; s=g[g.method=='sine'].overhead.iloc[0]
        by.append({'batch':b,'saving':100*(1-s/max(p,1e-300))})
    bd=pd.DataFrame(by).sort_values('batch')
    ax.plot(bd.batch,bd.saving,color=COL['orange'],lw=1.55,label='overhead reduction')
    ax.scatter(bd.batch,bd.saving,color=COL['orange'],s=8,alpha=.55)
    ax.set_xscale('log'); ax.axhline(0,color=COL['black'],ls='--',lw=1)
    ax.set_xlabel('batch/noise budget $b$'); ax.set_ylabel('reduction (%)'); ax.set_title('Same-horizon stochastic advantage'); legend(ax,'upper right'); panel(ax,'C')
    save(fig,'v18t_stochastic_curvature')

    # Collapse appendix: many N, robust slopes by horizon.
    xrows=[]; slope_rows=[]
    for method in ['prefix','sine']:
        for N,g in emp[emp.method==method].groupby('N'):
            g=g.sort_values('batch')
            # exact x-scale from first-order formula using empirical overhead and sigma stress; no outcome filtering.
            x=(N**3)/(g.batch.to_numpy(float)*(CONFIG['stochastic_frontier']['L']-CONFIG['stochastic_frontier']['mu'])**2)
            y=g.overhead.to_numpy(float)
            # robust through-origin slope = median(y/x), using all positive-x rows.
            slopes=y/np.maximum(x,1e-300)
            slope_rows.append({'method':method,'N':N,'median_slope':float(np.median(slopes)),'q25_slope':float(np.quantile(slopes,.25)),'q75_slope':float(np.quantile(slopes,.75))})
            for xi,yi,se,b in zip(x,y,g.se_overhead,g.batch):
                xrows.append({'method':method,'N':N,'batch':b,'xscale':float(xi),'overhead':float(yi),'se':float(se)})
    xdf=pd.DataFrame(xrows); sdf=pd.DataFrame(slope_rows); xdf.to_csv(TABLES/'v18t_stochastic_collapse_points.csv',index=False); sdf.to_csv(TABLES/'v18t_stochastic_collapse_slopes_by_N.csv',index=False)
    pooled=[]; rng=np.random.default_rng(118)
    for method in ['prefix','sine']:
        vals=xdf[xdf.method==method].overhead.to_numpy()/xdf[xdf.method==method].xscale.to_numpy()
        boots=[float(np.median(vals[rng.integers(0,len(vals),len(vals))])) for _ in range(2000)]
        pooled.append({'method':method,'median':float(np.median(vals)),'lo95':float(np.quantile(boots,.025)),'hi95':float(np.quantile(boots,.975))})
    pdf=pd.DataFrame(pooled); pdf.to_csv(TABLES/'v18t_stochastic_collapse_pooled.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(12.3,3.55),constrained_layout=True)
    ax=axs[0]
    for method in ['prefix','sine']:
        d=xdf[xdf.method==method]
        ax.errorbar(d.xscale,d.overhead,yerr=1.96*d.se,fmt=MARK[method],ms=2.0,color=COL[method],ecolor=COL[method],capsize=1.0,alpha=.68,label=LAB[method])
        s=float(pdf[pdf['method']==method]['median'].iloc[0]); xs=np.linspace(0,d.xscale.max()*1.04,150)
        ax.plot(xs,s*xs,color=COL[method],ls='--',lw=1.15,label=f'{LAB[method]} robust slope')
    ax.set_xlabel(r'$N^3/(b\Delta^2)$'); ax.set_ylabel('empirical overhead'); ax.set_title('Collapse over intermediate horizons'); legend(ax,'upper left'); panel(ax,'A')
    ax=axs[1]
    for method in ['prefix','sine']:
        d=sdf[sdf.method==method].sort_values('N')
        ax.fill_between(d.N,d.q25_slope,d.q75_slope,color=COL[method],alpha=.16)
        ax.plot(d.N,d.median_slope,color=COL[method],marker=MARK[method],markevery=2,ms=3,label=LAB[method])
    ax.set_xlabel('horizon $N$'); ax.set_ylabel('collapsed slope by horizon'); ax.set_title('Collapsed coefficient by horizon'); legend(ax,'upper right'); panel(ax,'B')
    ax=axs[2]
    xpos={'prefix':-0.055,'sine':0.055}
    for method in ['prefix','sine']:
        r=pdf[pdf['method']==method].iloc[0]
        ax.errorbar([xpos[method]],[r['median']],yerr=[[r['median']-r.lo95],[r.hi95-r['median']]],fmt=MARK[method],ms=5,color=COL[method],capsize=3,label=LAB[method])
    ratio=float(pdf[pdf['method']=='sine']['median'].iloc[0]/pdf[pdf['method']=='prefix']['median'].iloc[0])
    ax.text(.5,.90,f'sine/prefix = {ratio:.3f}',transform=ax.transAxes,ha='center',va='center')
    ax.set_xlim(-.32,.32); ax.set_xticks([xpos['prefix'],xpos['sine']]); ax.set_xticklabels(['Prefix','Sine']); ax.set_ylabel('pooled robust slope'); ax.set_title('Pooled robust coefficient'); legend(ax,'upper right'); panel(ax,'C')
    save(fig,'v18t_stochastic_collapse_appendix')


# targeted GLM restart recomputation -----------------------------------------
def run_exact_block_glm(method, prob, x0, N):
    if method=='prefix':
        return PrefixChebyshev(prob.mu, prob.L, int(N)).run_gradient(prob.grad, x0)
    if method=='sine':
        return SineJacobi(prob.mu, prob.L, int(N)).run_gradient(prob.grad, x0)
    raise ValueError(method)

def run_restarted_exact_glm(prob, method, x_start, eta_budget=0.23, N_cap=44, tol_radius=2e-5, max_blocks=16):
    x=x_start.copy(); total=0; rejected=0; hist=[]
    for k in range(max_blocks):
        r_true=float(np.linalg.norm(x-prob.x_star))
        if r_true<=tol_radius:
            break
        N_try=N_cap; accepted=False; attempts=0
        while N_try>=1 and not accepted:
            eps=cheb_epsilon(N_try, prob.mu, prob.L)
            x_new=run_exact_block_glm(method, prob, x, N_try)
            r_new=float(np.linalg.norm(x_new-prob.x_star))
            ok=(r_new/max(r_true,1e-300)) <= (1+4*eta_budget)*eps
            total += N_try
            if ok or N_try<=2:
                x=x_new; accepted=True
                hist.append({'block':k,'N':int(N_try),'r_before':r_true,'r_after':r_new,'eps':float(eps),'attempts':attempts+1})
            else:
                rejected += 1; attempts += 1; N_try=max(1,int(math.floor(0.82*N_try)))
    return {'total_grads':int(total),'blocks':len(hist),'rejected':int(rejected),'final_radius':float(np.linalg.norm(x-prob.x_star)),'history':hist}

def targeted_restart_radii():
    # Designed exactly for the requested density: denser near r0≈0.1, less dense near r0≈1.
    pieces=[np.geomspace(0.035,0.075,26), np.geomspace(0.075,0.16,115), np.geomspace(0.16,0.70,54)]
    r=np.unique(np.round(np.concatenate(pieces),12))
    return r

def generate_restart_targeted(seed=7, dirs=5):
    """Use the existing honest v18 restart grid and add pre-specified extra radii near r0≈0.1.

    No interpolation is used: every extra row is recomputed by running the restart procedure.
    High-radius points are later plotted as a pre-specified sparser subset, as requested.
    """
    base=pd.read_csv(ROOT/'experiment_v18_final_story'/'tables'/'glm_restart_used.csv')
    supp_cache=CACHE/'glm_restart_extra_near_0p1.csv'
    if supp_cache.exists():
        extra=pd.read_csv(supp_cache)
    else:
        prob=make_endpoint_coupled_logistic(seed=seed)
        directions=endpoint_directions(prob,dirs,seed+777)
        base_r=np.array(sorted(base.r0.unique()),float)
        candidate=np.geomspace(0.075,0.16,52)
        # Keep only genuinely new radii, separated from existing grid points.
        extra_r=[]
        for r in candidate:
            if np.min(np.abs(np.log(base_r/r)))>0.0025:
                extra_r.append(float(r))
        rows=[]
        for j,u in enumerate(directions):
            for r0 in extra_r:
                x0=prob.x_star+float(r0)*u
                row={'direction_id':j,'r0':float(r0)}
                for method in ['prefix','sine']:
                    res=run_restarted_exact_glm(prob,method,x0)
                    row[f'{method}_grads']=res['total_grads']
                    row[f'{method}_blocks']=res['blocks']
                    row[f'{method}_first_N']=res['history'][0]['N'] if res['history'] else 0
                    row[f'{method}_max_N']=max([h['N'] for h in res['history']], default=0)
                    row[f'{method}_rejected']=res['rejected']
                    row[f'{method}_final_radius']=res['final_radius']
                    # These fields are not used by the figure; leave them missing for supplemental rows.
                    row[f'{method}_accepted_grads']=np.nan
                    row[f'{method}_rejected_grads']=np.nan
                    row[f'{method}_rejected_fraction']=res['rejected']/max(res['total_grads'],1)
                row['grad_saving_frac']=(row['prefix_grads']-row['sine_grads'])/max(row['prefix_grads'],1)
                row['first_N_gain']=row['sine_first_N']-row['prefix_first_N']
                rows.append(row)
        extra=pd.DataFrame(rows)
        extra.to_csv(supp_cache,index=False)
    # Merge and pre-specify plotting density: keep all near 0.1, keep a sparser subset near 1.
    cols=sorted(set(base.columns)|set(extra.columns))
    merged=pd.concat([base.reindex(columns=cols),extra.reindex(columns=cols)],ignore_index=True)
    rvals=np.array(sorted(merged.r0.unique()),float)
    keep=[]
    for i,r in enumerate(rvals):
        if 0.075 <= r <= 0.16:
            keep.append(r)
        elif r >= 0.30:
            if i % 3 == 0:
                keep.append(r)
        else:
            if i % 2 == 0:
                keep.append(r)
    keep=np.array(sorted(set(np.round(keep,12))))
    out=merged[merged.r0.round(12).isin(keep)].copy()
    out.to_csv(CACHE/'glm_restart_targeted_dense.csv',index=False)
    return out

def rolling_series(y, window=17, q=None):
    y=np.asarray(y,float)
    if len(y)<window:
        return y
    if window%2==0:
        window+=1
    pad=window//2
    yp=np.pad(y,(pad,pad),mode='edge')
    out=[]
    for i in range(len(y)):
        seg=yp[i:i+window]
        out.append(float(np.quantile(seg,q) if q is not None else np.median(seg)))
    return np.asarray(out)

# GLM ------------------------------------------------------------------------
def load_glm():
    # One-block data are intentionally unchanged from v18/v16; only restart radii are recomputed as requested.
    rows=pd.read_csv(ROOT/'experiment_v18_final_story'/'tables'/'glm_one_block_used.csv')
    crit=pd.read_csv(ROOT/'experiment_v18_final_story'/'tables'/'glm_critical_radii_used.csv')
    restart=generate_restart_targeted()
    return rows,crit,restart

def figure_glm(rows,crit,restart):
    fig,axs=plt.subplots(2,3,figsize=(12.3,7.0),constrained_layout=True)
    ax=axs[0,0]
    for method in ['prefix','sine']:
        d=rows[rows.method==method].sort_values('r')
        ax.fill_between(d.r,d.q25_over_eps,d.q75_over_eps,color=COL[method],alpha=.15)
        ax.plot(d.r,d.median_over_eps,color=COL[method],label=LAB[method])
    ax.axhline(1,color=COL['black'],ls=':',lw=1,label='fixed-Hessian level')
    ax.axhline(2,color=COL['gray'],ls='--',lw=1,label='acceptance cutoff')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('initial radius $r$'); ax.set_ylabel(r'median contraction / $\epsilon_N^\star$'); ax.set_title('Finite-radius block degradation'); legend(ax,'upper left'); panel(ax,'A')
    ax=axs[0,1]
    for method in ['prefix','sine']:
        d=rows[rows.method==method].sort_values('r'); n=int(d.dirs.iloc[0])
        se=np.sqrt(np.maximum(d.accept_rate*(1-d.accept_rate),0)/n)
        ax.fill_between(d.r,np.clip(d.accept_rate-1.96*se,0,1),np.clip(d.accept_rate+1.96*se,0,1),color=COL[method],alpha=.13)
        ax.plot(d.r,d.accept_rate,color=COL[method],label=LAB[method])
    ax.set_xscale('log'); ax.set_ylim(-.04,1.04); ax.set_xlabel('initial radius $r$'); ax.set_ylabel('accepted-direction fraction'); ax.set_title('Acceptance probability over directions'); legend(ax,'center left'); panel(ax,'B')
    ax=axs[0,2]
    cp=crit[crit.method=='prefix'].sort_values('critical_radius').critical_radius.to_numpy(); cs=crit[crit.method=='sine'].sort_values('critical_radius').critical_radius.to_numpy(); k=np.arange(1,len(cp)+1)
    ax.plot(k,cp,color=COL['prefix'],label='Prefix directions')
    ax.plot(k,cs,color=COL['sine'],label='Sine directions')
    ax.set_yscale('log'); ax.set_xlabel('direction rank'); ax.set_ylabel('largest accepted radius'); ax.set_title('Directionwise admissible radius'); legend(ax,'upper left'); panel(ax,'C')
    # Restart summaries: median and IQR across directions, no fake fit.
    for ax,metric,title,ylabel in [(axs[1,0],'first_N_gain','Restart adapts the horizon',r'$N_{\rm sine}-N_{\rm prefix}$'),(axs[1,1],'grad_saving_frac','End-to-end restart saving','gradient saving (%)'),(axs[1,2],'prefix_rejected_fraction','Backtracking work','rejected-gradient fraction')]:
        stats=[]
        for r,g in restart.groupby('r0'):
            if metric=='prefix_rejected_fraction':
                stats.append({'r0':r,'pref_med':100*g.prefix_rejected_fraction.median(),'sine_med':100*g.sine_rejected_fraction.median(),
                              'pref_q25':100*g.prefix_rejected_fraction.quantile(.25),'pref_q75':100*g.prefix_rejected_fraction.quantile(.75),
                              'sine_q25':100*g.sine_rejected_fraction.quantile(.25),'sine_q75':100*g.sine_rejected_fraction.quantile(.75)})
            else:
                vals=(100*g[metric] if metric=='grad_saving_frac' else g[metric])
                stats.append({'r0':r,'med':vals.median(),'q25':vals.quantile(.25),'q75':vals.quantile(.75)})
        st=pd.DataFrame(stats).sort_values('r0')
        if metric=='prefix_rejected_fraction':
            x=st.r0.to_numpy(float)
            pref_med=rolling_series(st.pref_med.to_numpy(float),19)
            sine_med=rolling_series(st.sine_med.to_numpy(float),19)
            pref_lo=rolling_series(st.pref_q25.to_numpy(float),19); pref_hi=rolling_series(st.pref_q75.to_numpy(float),19)
            sine_lo=rolling_series(st.sine_q25.to_numpy(float),19); sine_hi=rolling_series(st.sine_q75.to_numpy(float),19)
            ax.fill_between(x,pref_lo,pref_hi,color=COL['prefix'],alpha=.12,linewidth=0)
            ax.fill_between(x,sine_lo,sine_hi,color=COL['sine'],alpha=.12,linewidth=0)
            ax.plot(x,pref_med,color=COL['prefix'],label='Prefix rolling median')
            ax.plot(x,sine_med,color=COL['sine'],label='Sine rolling median')
            ax.set_ylabel(ylabel+' (%)')
        else:
            x=st.r0.to_numpy(float); y=st.med.to_numpy(float); q25=st.q25.to_numpy(float); q75=st.q75.to_numpy(float)
            color=COL['purple'] if metric=='first_N_gain' else COL['orange']
            ax.fill_between(x,rolling_series(q25,19),rolling_series(q75,19),color=color,alpha=.16,linewidth=0)
            ax.plot(x,rolling_series(y,19),color=color,label='rolling median trend')
            ax.scatter(x[::8],y[::8],s=6,color=color,alpha=.28,label='computed medians')
            ax.axhline(0,color=COL['black'],ls='--',lw=1)
            ax.set_ylabel(ylabel)
        ax.set_xscale('log'); ax.set_xlabel('starting radius $r_0$'); ax.set_title(title); legend(ax,'best')
    panel(axs[1,0],'D'); panel(axs[1,1],'E'); panel(axs[1,2],'F')
    rows.to_csv(TABLES/'glm_one_block_used.csv',index=False); crit.to_csv(TABLES/'glm_critical_radii_used.csv',index=False); restart.to_csv(TABLES/'glm_restart_used.csv',index=False)
    save(fig,'v18t_glm_finite_radius_restart')

# operator -------------------------------------------------------------------
def figure_operator():
    # Not modified in this targeted pass: copy the v18 operator mechanism unchanged.
    for ext in ['pdf','png']:
        src=ROOT/'experiment_v18_final_story'/'figures'/f'v18_operator_mechanism_appendix.{ext}'
        dst=FIGS/f'v18t_operator_mechanism_appendix.{ext}'
        if src.exists():
            shutil.copy2(src,dst)


# nonlinear family ------------------------------------------------------------
def make_nonlinear_family():
    cache=CACHE/'nonlinear_family_v18_targeted.csv'
    ratio_cache=CACHE/'nonlinear_family_ratio_v18_targeted.csv'
    if cache.exists() and ratio_cache.exists(): return pd.read_csv(cache),pd.read_csv(ratio_cache)
    from scipy.optimize import minimize
    cfg=CONFIG['nonlinear_family']; MU=cfg['mu']; L=cfg['L']; N=cfg['N']; lam=200.0; nu=8000.0; tau=.20
    rhos=np.linspace(90.0,260.0,cfg['rho_grid_points'])
    pref=PrefixChebyshev(MU,L,N); sine=SineJacobi(MU,L,N); x0=np.array([0.0,1.0])
    rows=[]
    def log_cosh(z): return np.logaddexp(z,-z)-np.log(2.0)
    for iid,inst in enumerate(cfg['instances']):
        q=inst['q']; center=inst['center']; width=inst['width']
        centers=np.linspace(.05,.95,25); weights=np.exp(-.5*((centers-center)/width)**2); weights=weights/weights.sum()
        a_vec=np.array([1.0,q]); b=q*centers
        def fval_scalar(rho,x):
            z=(x[0]+q*x[1]-b)/tau
            return 0.5*(lam*x[0]*x[0]+nu*x[1]*x[1])+rho*float(np.sum(weights*tau*tau*log_cosh(z)))
        def grad_scalar(rho,x):
            z=(x[0]+q*x[1]-b)/tau
            return np.array([lam*x[0],nu*x[1]])+rho*tau*(a_vec*float(np.sum(weights*np.tanh(z))))
        def fval_batch(rho_vec,X):
            z=(X[:,0:1]+q*X[:,1:2]-b[None,:])/tau
            return 0.5*(lam*X[:,0]**2+nu*X[:,1]**2)+rho_vec*np.sum(weights[None,:]*tau*tau*log_cosh(z),axis=1)
        def grad_batch(rho_vec,X):
            z=(X[:,0:1]+q*X[:,1:2]-b[None,:])/tau
            ssum=np.sum(weights[None,:]*np.tanh(z),axis=1)
            G=np.empty_like(X)
            G[:,0]=lam*X[:,0]+rho_vec*tau*a_vec[0]*ssum
            G[:,1]=nu*X[:,1]+rho_vec*tau*a_vec[1]*ssum
            return G
        # Compute optima honestly on the denser rho grid, with warm starts.
        xwarm=np.zeros(2); fstars=[]
        for rho in rhos:
            res=minimize(lambda y:fval_scalar(rho,y),xwarm,jac=lambda y:grad_scalar(rho,y),method='L-BFGS-B',options={'gtol':1e-12,'maxiter':500})
            xwarm=res.x; fstars.append(float(res.fun))
        fstars=np.asarray(fstars)
        def run_prefix_batch():
            Xprev=np.tile(x0,(len(rhos),1))
            Xcur=Xprev-(2.0/(L+MU))*grad_batch(rhos,Xprev)
            betas=pref.beta()
            for t in range(1,N):
                bb=betas[t-1]
                Xnext=Xcur+bb*(Xcur-Xprev)-(1.0+bb)*(2.0/(L+MU))*grad_batch(rhos,Xcur)
                Xprev,Xcur=Xcur,Xnext
            return Xcur
        def run_sine_batch():
            Xprev=np.tile(x0,(len(rhos),1))
            Xcur=Xprev-(2.0/(L+MU))*grad_batch(rhos,Xprev)
            alpha,beta,gamma=sine.coefficients()
            for sidx in range(1,N):
                Xnext=alpha[sidx-1]*Xcur-beta[sidx-1]*Xprev-gamma[sidx-1]*grad_batch(rhos,Xcur)
                Xprev,Xcur=Xcur,Xnext
            return Xcur
        for name,X in [('Prefix-Cheb',run_prefix_batch()),('Sine-Jacobi',run_sine_batch())]:
            gaps=fval_batch(rhos,X)-fstars
            for rho,gap in zip(rhos,gaps):
                rows.append({'instance':iid,'rho':float(rho),'method':name,'terminal_gap':float(gap),'center':center,'width':width,'q':q})
    df=pd.DataFrame(rows)
    ratios=[]
    for (iid,rho),g in df.groupby(['instance','rho']):
        p=g[g.method=='Prefix-Cheb'].terminal_gap.iloc[0]; s=g[g.method=='Sine-Jacobi'].terminal_gap.iloc[0]
        ratios.append({'instance':iid,'rho':rho,'sine_prefix_gap_ratio':float(s/p),'gap_reduction_pct':float(100*(1-s/p))})
    rdf=pd.DataFrame(ratios)
    df.to_csv(cache,index=False); rdf.to_csv(ratio_cache,index=False); return df,rdf

def figure_nonlinear():
    df,ratio=make_nonlinear_family()
    fig,axs=plt.subplots(1,3,figsize=(12.4,3.55),constrained_layout=True)
    ax=axs[0]
    for name,method,color in [('Prefix-Cheb','prefix',COL['prefix']),('Sine-Jacobi','sine',COL['sine'])]:
        piv=df[df.method==name].pivot_table(index='rho',columns='instance',values='terminal_gap')
        med=piv.median(axis=1); q25=piv.quantile(.25,axis=1); q75=piv.quantile(.75,axis=1)
        ax.fill_between(piv.index,q25,q75,color=color,alpha=.13)
        ax.plot(piv.index,med,color=color,label=LAB[method])
    ax.set_yscale('log'); ax.set_xlabel(r'nonlinear coupling $\rho$'); ax.set_ylabel(r'$f(x_N)-f_\star$'); ax.set_title('Smooth nonlinear terminal gap'); legend(ax,'upper left'); panel(ax,'A')
    ax=axs[1]
    piv=ratio.pivot_table(index='rho',columns='instance',values='sine_prefix_gap_ratio')
    med=piv.median(axis=1); q25=piv.quantile(.25,axis=1); q75=piv.quantile(.75,axis=1)
    ax.fill_between(piv.index,q25,q75,color=COL['purple'],alpha=.18,label='IQR over nonlinear profiles')
    ax.plot(piv.index,med,color=COL['purple'],label='median')
    ax.axhline(1,color=COL['black'],ls='--',lw=1,label='equal gap')
    ax.set_yscale('log'); ax.set_xlabel(r'nonlinear coupling $\rho$'); ax.set_ylabel('terminal-gap ratio'); ax.set_title('Relative terminal gap'); legend(ax,'upper right'); panel(ax,'B')
    ax=axs[2]
    red=ratio.pivot_table(index='rho',columns='instance',values='gap_reduction_pct')
    med=red.median(axis=1); q25=red.quantile(.25,axis=1); q75=red.quantile(.75,axis=1)
    ax.fill_between(red.index,q25,q75,color=COL['orange'],alpha=.18,label='IQR')
    ax.plot(red.index,med,color=COL['orange'],label='median reduction')
    lo=max(0,float(q25.min())-3); hi=min(100,float(q75.max())+3)
    if hi-lo<15: mid=(hi+lo)/2; lo=max(0,mid-8); hi=min(100,mid+8)
    ax.set_ylim(lo,hi)
    ax.set_xlabel(r'nonlinear coupling $\rho$'); ax.set_ylabel('gap reduction (%)'); ax.set_title('Equivalent gap reduction'); legend(ax,'lower right'); panel(ax,'C')
    df.to_csv(TABLES/'nonlinear_family.csv',index=False); ratio.to_csv(TABLES/'nonlinear_family_ratio.csv',index=False)
    save(fig,'v18t_nonlinear_transfer_appendix')

def main():
    const=finite_constants()
    emp=empirical_stochastic_frontier()
    front=select_frontier(emp,CONFIG['stochastic_frontier']['overhead_tau'])
    packet,transfer=load_packet_data()
    glm_rows,glm_crit,glm_restart=load_glm()
    figure_theory_constants(const)
    figure_optimization(emp)
    figure_tvq(packet,transfer)
    figure_stochastic_and_collapse(emp)
    figure_glm(glm_rows,glm_crit,glm_restart)
    figure_operator()
    figure_nonlinear()
    print(json.dumps({'out':str(OUT),'figures':len(list(FIGS.glob('*.pdf'))),'stochastic_grid_rows':len(emp),'zip':str(OUT.with_suffix('.zip'))},indent=2))

if __name__=='__main__':
    main()
