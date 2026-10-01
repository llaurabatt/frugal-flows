"""Overnight batch runner.   usage: overnight.py <setup> <seed_fit> [--smoke]

Data seed fixed at 101 throughout; only seed_fit varies, so replicates differ in
initialisation and batching alone. The train/validation split is the canonical one
(reproduced from the seed-101 walk the first margin-only fit used) and is identical in
every run. Checkpoint selection is always ordinary UNWEIGHTED held-out NLL.
Stopping: patience 30, epoch cap 1000, wall-clock cap 2h; the reason is recorded, and a
time- or epoch-limited fit is marked as unfinished rather than converged.
"""
import os, sys, time, json, math, numpy as np, jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx, optax, paramax
sys.path.insert(0, "/home/llaurabat/ff-project/frugal-flows/validation/morphomnist")
jax.config.update("jax_enable_x64", False)
from flowjax.bijections import Invert, Stack, Tanh
from flowjax.distributions import Transformed, Uniform
import frugal_flows.causal_flows as CF
from frugal_flows.causal_flows import _build_flexible_margin, get_independent_quantiles
import exp_ate_recovery as E

SP="/tmp/claude-2002/-home-llaurabat-ff-project/7ae089eb-8ecf-45ac-98e7-c90c357166d0/scratchpad"; ROOT=f"{SP}/overnight"
SETUPS={
 "zero_mlp_8":  dict(preset="exp1_rct_homogeneous",size=8, base_shift=0.0, model="standalone",cond="mlp",        lr=1e-2),
 "ref_mlp_8":   dict(preset="exp1_rct_homogeneous",size=8, base_shift=None,model="standalone",cond="mlp",        lr=1e-2),
 "sep_mlp_8":   dict(preset="exp1_rct_homogeneous",size=8, base_shift=None,model="separate",  cond="mlp",        lr=1e-2),
 "std_trf_16":  dict(preset="exp1_rct_homogeneous",size=16,base_shift=None,model="standalone",cond="transformer",lr=1e-3),
 "ff_trf_16_e1":dict(preset="exp1_rct_homogeneous",size=16,base_shift=None,model="fullff",    cond="transformer",lr=1e-3),
 "ff_trf_16_e2":dict(preset="exp2_confounded_homogeneous",size=16,base_shift=None,model="fullff",cond="transformer",lr=1e-3),
}
setup=sys.argv[1]; SEED_FIT=int(sys.argv[2]); SMOKE="--smoke" in sys.argv; TIME="--time" in sys.argv; S=SETUPS[setup]
SEED_DATA,PATIENCE,EPOCH_CAP,WALL_CAP=101,30,1000,2*3600
TRACK_EVERY,NMC_TRACK,NMC_FINAL,NMC_SEP=20,2000,5000,20000
if SMOKE: EPOCH_CAP,WALL_CAP,TRACK_EVERY,NMC_TRACK,NMC_FINAL,NMC_SEP,PATIENCE=4,1200,2,200,300,600,99
if TIME:  EPOCH_CAP,WALL_CAP,TRACK_EVERY,NMC_TRACK,NMC_FINAL,NMC_SEP,PATIENCE=3,3600,2,500,500,1000,99
TAG=f"{setup}_s{SEED_FIT}"+("_smoke" if SMOKE else "")+("_time" if TIME else ""); OUT=f"{ROOT}/{TAG}"; os.makedirs(OUT,exist_ok=True)
MARGIN=dict(RQS_knots=8,nn_depth=1,nn_width=48,flow_layers=4,conditioner=S["cond"],nn_heads=4)

kw=dict(preset=S["preset"],arm="flexible_continuous",size=S["size"],seed_data=SEED_DATA,seed_fit=SEED_FIT)
if S["base_shift"] is not None: kw["base_shift"]=S["base_shift"]
if SMOKE: kw["n"]=600
cfg=E.Config(**kw); data=E.build_data(cfg)
Yv=np.asarray(data["Y"],np.float32); Xv=np.asarray(data["X"],np.float32)
ITE=np.asarray(data["ITE"]); ate=np.asarray(data["ATE"])
Y0=Yv-Xv[:,:1]*ITE; Y1=Y0+ITE; n,K,s=Yv.shape[0],Yv.shape[1],S["size"]
if S["base_shift"]==0.0:
    assert np.abs(Y1-Y0).max()==0 and np.abs(ate).max()==0, "base-shift 0 must give identical potential outcomes"
    print("  verified: Y(1) == Y(0) exactly, true effect zero at every pixel", flush=True)

radius=round(s/4); xx,yy=np.meshgrid(np.arange(s),np.arange(s),indexing="ij"); c=(s-1)/2
disc=(((xx-c)**2+(yy-c)**2)<=radius**2); ring=np.zeros_like(disc)
for i in range(s):
    for j in range(s):
        if disc[i,j]: continue
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ii,jj=i+di,j+dj
            if 0<=ii<s and 0<=jj<s and disc[ii,jj]: ring[i,j]=True
REG={"reference_disc":disc.ravel(),"reference_ring":ring.ravel(),"far_region":(~disc&~ring).ravel()}

k0=jr.PRNGKey(101)
for _ in range(3): k0,_=jr.split(k0)
_,sk=jr.split(k0); perm=np.asarray(jr.permutation(sk,jnp.arange(n))); ntr=n-round(0.1*n)
TR,VA=perm[:ntr],perm[ntr:]; np.savez(f"{OUT}/split.npz",train=TR,val=VA)

from flowjax.bijections import MaskedAutoregressive, RationalQuadraticSpline, Scan
from flowjax.flows import _add_default_permute
def uncond_margin_bijection(key,dim,RQS_knots,nn_depth,nn_width,flow_layers):
    """masked_autoregressive_bijection with cond_dim=None. The library wrapper reads
    condition.shape[1] unconditionally, so it cannot build an unconditional margin; this
    mirrors it line for line with the treatment input removed. Same spline, same layer
    count, same permutations, same inversion."""
    tr=RationalQuadraticSpline(knots=RQS_knots,interval=1)
    def make_layer(k):
        bk,pk=jr.split(k)
        b=MaskedAutoregressive(key=bk,transformer=tr,dim=dim,cond_dim=None,nn_width=nn_width,nn_depth=nn_depth)
        return _add_default_permute(b,dim,pk)
    return Invert(Scan(eqx.filter_vmap(make_layer)(jr.split(key,flow_layers))))

def margin_dist(key,cond):
    if cond is None:
        m=uncond_margin_bijection(key,K,MARGIN["RQS_knots"],MARGIN["nn_depth"],MARGIN["nn_width"],MARGIN["flow_layers"])
        d=Transformed(Uniform(-jnp.ones(K),jnp.ones(K)),m)
        return Transformed(d,Stack([Invert(Tanh(()))]*K)).merge_transforms()
    m=_build_flexible_margin(key=key,dim=K,condition=cond,causal_model_args=MARGIN)
    d=Transformed(Uniform(-jnp.ones(K),jnp.ones(K)),m)
    return Transformed(d,Stack([Invert(Tanh(()))]*K)).merge_transforms()
def npar(t): return sum(int(x.size) for x in jax.tree_util.tree_leaves(eqx.filter(t,eqx.is_inexact_array)))

# ---------------- evaluation ----------------
def acct(y,name):
    nan=int(np.isnan(y).sum()); inf=int(np.isinf(y).sum()); fin=y[np.isfinite(y)]
    return {f"{name}_nan_values":nan,f"{name}_inf_values":inf,
            f"{name}_max_abs_finite":float(np.abs(fin).max()) if fin.size else float("nan")}
def evaluate(y0,y1,tgt0,tgt1,paired,prefix=""):
    m={}; m.update(acct(y0,prefix+"y0")); m.update(acct(y1,prefix+"y1"))
    nonfinite=m[prefix+"y0_nan_values"]+m[prefix+"y0_inf_values"]+m[prefix+"y1_nan_values"]+m[prefix+"y1_inf_values"]
    m[prefix+"nonfinite_total"]=nonfinite; m[prefix+"ate_valid"]=(nonfinite==0)
    tau=(y1-y0).mean(0)                                    # PRIMARY: every draw kept
    err=tau-ate
    m[prefix+"ate_mae"]=float(np.abs(err).mean()); m[prefix+"ate_rmse"]=float(np.sqrt((err**2).mean()))
    for nm,msk in REG.items():
        d=(y1[:,msk]-y0[:,msk]).mean(1) if paired else None
        a=y1[:,msk].mean(1); b=y0[:,msk].mean(1)
        est=float(a.mean()-b.mean()); truth=float(ate[msk].mean())
        m[f"{prefix}{nm}_signed_err"]=est-truth
        m[f"{prefix}{nm}_mae"]=float(np.abs(err[msk]).mean())
        m[f"{prefix}{nm}_se_unpaired"]=float(np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)))
        if paired: m[f"{prefix}{nm}_se_paired"]=float(d.std(ddof=1)/np.sqrt(len(d)))
        m[f"{prefix}{nm}_e0"]=float(b.mean()-tgt0[msk].mean()); m[f"{prefix}{nm}_e1"]=float(a.mean()-tgt1[msk].mean())
    keep=np.isfinite(y0).all(1)&np.isfinite(y1).all(1)
    if nonfinite:
        tf=(y1[keep]-y0[keep]).mean(0)
        m[prefix+"DIAGNOSTIC_finite_only_ate_mae"]=float(np.abs(tf-ate).mean())
        m[prefix+"DIAGNOSTIC_draws_excluded"]=int((~keep).sum())
    d0=np.abs(y1-y0).max(1); cut=np.quantile(d0,0.999)      # influence of extreme finite draws, no clipping
    sel=d0<=cut
    m[prefix+"extreme_draw_influence_on_mae"]=float(np.abs((y1[sel]-y0[sel]).mean(0)-ate).mean()-m[prefix+"ate_mae"])
    return m, tau, y0.mean(0)-tgt0, y1.mean(0)-tgt1

def sample_cond(p,st,seed,n_mc,cond_dim,slice_K=False):
    d=eqx.combine(p,st)
    out=[]
    for t in (0.0,1.0):
        z=d.sample(jr.key(seed),condition=jnp.full((n_mc,cond_dim),t))
        z=np.asarray(z); out.append(z[:,:K] if slice_K else z)
    return out[0],out[1]
def sample_uncond(p0,st0,p1,st1,seed,n_mc):
    y0=np.asarray(eqx.combine(p0,st0).sample(jr.key(seed),sample_shape=(n_mc,)))
    y1=np.asarray(eqx.combine(p1,st1).sample(jr.key(seed),sample_shape=(n_mc,)))
    return y0,y1

# ---------------- training loop ----------------
def run_loop(dist,x,cond,tr,va,lr,track_fn=None,grad_fn=None,lam=1.0,label="",step0=0,pfx=""):
    params,static=eqx.partition(dist,eqx.is_inexact_array,is_leaf=lambda l:isinstance(l,paramax.NonTrainable))
    opt=optax.adam(lr); ost=opt.init(params)
    def terms(p,st,xb,cb):
        d=paramax.unwrap(eqx.combine(p,st)); total=d.log_prob(xb,cb)
        B=d.bijection.bijections; xs=xb; ld=jnp.zeros(xb.shape[0])
        for b in reversed(B[3:]):
            if b.cond_shape is not None: xs,l=jax.vmap(lambda xi,ci:b.inverse_and_log_det(xi,ci))(xs,cb)
            else:                        xs,l=jax.vmap(lambda xi:b.inverse_and_log_det(xi))(xs)
            ld=ld+l
        marg=ld-K*jnp.log(2.0); return total,marg,total-marg
    def unw(p,st,xb,cb):
        d=paramax.unwrap(eqx.combine(p,st)); return -d.log_prob(xb,cb).mean()
    def obj(p,st,xb,cb):
        if lam==1.0: return unw(p,st,xb,cb)
        _,mg,cp=terms(p,st,xb,cb); return -(mg+lam*cp).mean()
    @eqx.filter_jit
    def step(p,st,xb,cb,o):
        l,g=eqx.filter_value_and_grad(obj)(p,st,xb,cb); up,o=opt.update(g,o,p)
        return eqx.apply_updates(p,up),o,l
    @eqx.filter_jit
    def vnll(p,st,xb,cb): return unw(p,st,xb,cb)
    def bat(idx,key):
        idx=np.asarray(jr.permutation(key,jnp.asarray(idx))); nb=max(len(idx)//cfg.batch_size,1)
        return [idx[i*cfg.batch_size:(i+1)*cfg.batch_size] for i in range(nb)]
    best,bp,bep,hist,why=np.inf,params,0,[],"epoch_cap"
    track=TRACK_EVERY; sched=[]; t0=time.monotonic(); key=jr.PRNGKey(SEED_FIT); ep_t=[]
    for ep in range(1,EPOCH_CAP+1):
        te=time.monotonic(); key,bk=jr.split(key); tl=[]
        for b in bat(tr,bk):
            params,ost,l=step(params,static,x[b],None if cond is None else cond[b],ost); tl.append(float(l))
        vn=float(np.mean([float(vnll(params,static,x[b],None if cond is None else cond[b])) for b in bat(va,jr.PRNGKey(0))]))
        ep_t.append(time.monotonic()-te)
        row=dict(epoch=ep,train_obj=float(np.mean(tl)),val_nll=vn,elapsed_s=time.monotonic()-t0)
        if cond is not None:
            t_rows=[b for b in bat(tr,jr.PRNGKey(0)) if True]
            row["val_nll_treated"]=float(np.mean([float(vnll(params,static,x[b[Xv[b,0]==1]],cond[b[Xv[b,0]==1]])) for b in bat(va,jr.PRNGKey(0)) if (Xv[b,0]==1).sum()>1]))
            row["val_nll_untreated"]=float(np.mean([float(vnll(params,static,x[b[Xv[b,0]==0]],cond[b[Xv[b,0]==0]])) for b in bat(va,jr.PRNGKey(0)) if (Xv[b,0]==0).sum()>1]))
        if lam!=1.0:
            row["val_obj_weighted"]=float(np.mean([float(obj(params,static,x[b],None if cond is None else cond[b])) for b in bat(va,jr.PRNGKey(0))]))
        if vn<best:
            best,bp,bep=vn,params,ep; eqx.tree_serialise_leaves(f"{OUT}/best{label}.eqx",eqx.combine(params,static))
        if ep%track==0 or ep==1:
            tt=time.monotonic()
            if track_fn: row.update(track_fn(params,static))
            if grad_fn:  row.update(grad_fn(params,static,terms))
            dt=time.monotonic()-tt; sched.append(ep)
            if ep==1 and np.mean(ep_t)>0 and dt>0.15*track*np.mean(ep_t):
                track=int(track*math.ceil(dt/(0.15*track*np.mean(ep_t))))
                print(f"    diagnostics cost {dt:.0f}s vs {np.mean(ep_t):.1f}s/epoch -> tracking every {track} epochs",flush=True)
        hist.append(row)
        if RUN is not None: RUN.log({pfx+k:v for k,v in row.items()},step=step0+ep)
        if ep%10==0: print(f"    ep {ep:>4} val {vn:8.3f} (best {best:8.3f} @ {bep})  {time.monotonic()-t0:.0f}s",flush=True)
        if time.monotonic()-t0>WALL_CAP: why="wall_clock"; break
        if ep-bep>=PATIENCE:               why="patience";   break
    eqx.tree_serialise_leaves(f"{OUT}/last{label}.eqx",eqx.combine(params,static))
    return bp,static,hist,dict(termination=why,best_val_nll=best,best_epoch=bep,epochs_run=len(hist),
                               wall_s=time.monotonic()-t0,track_schedule=sched,converged=(why=="patience"))

# ---------------- main ----------------
import wandb
RUN=wandb.init(entity="proj-lb",project="Frugal Images",group="overnight_batch",name=TAG,reinit=True,
               tags=["overnight",setup,S["preset"][:4],f"k{K}",S["cond"],("smoke" if SMOKE else "real")],
               config=dict(setup=setup,model=S["model"],conditioner=S["cond"],preset=S["preset"],size=s,K=K,
                           base_shift=S["base_shift"],learning_rate=S["lr"],seed_data=SEED_DATA,seed_fit=SEED_FIT,
                           patience=PATIENCE,epoch_cap=EPOCH_CAP,wall_cap_s=WALL_CAP,margin=MARGIN,
                           n_units=n,n_train=len(TR),n_val=len(VA),track_every=TRACK_EVERY,
                           nmc_track=NMC_TRACK,nmc_final=NMC_FINAL,nmc_separate=NMC_SEP,smoke=SMOKE))
t_all=time.monotonic(); res={}
print(f"### {TAG}: {S['model']} / {S['cond']} / {S['preset'][:4]} / K={K} / lr={S['lr']} / fit seed {SEED_FIT}",flush=True)

if S["model"] in ("standalone","fullff"):
    if S["model"]=="standalone":
        dist=margin_dist(jr.PRNGKey(SEED_FIT),jnp.asarray(Xv)); x=jnp.asarray(Yv); cond=jnp.asarray(Xv); sl=False; gfn=None
    else:
        uz=None; p=f"{ROOT}/_uz_{S['preset'][:4]}_k{K}{'_smoke' if SMOKE else ''}.npy"
        if os.path.exists(p): uz=np.load(p); print("  reusing the cached thickness marginal",flush=True)
        else:
            r=get_independent_quantiles(key=jr.PRNGKey(0),z_cont=jnp.asarray(data["z_cont"]),
                                        max_epochs=cfg.marginal_max_epochs,max_patience=cfg.marginal_max_patience,
                                        return_z_cont_flow=True)
            uz=np.asarray(r["u_z_cont"]); np.save(p,uz); print("  fitted and cached the thickness marginal",flush=True)
        cap={}
        def _cap(key,dist,data,**kwargs): cap["d"]=dist; return dist,{"train":[0.],"val":[0.]}
        _orig=CF.fit_to_data; CF.fit_to_data=_cap
        CF.train_frugal_flow(causal_model="flexible_continuous",key=jr.PRNGKey(SEED_FIT),y=jnp.asarray(Yv),
                             u_z=jnp.asarray(uz),condition=jnp.asarray(Xv),causal_model_args=dict(MARGIN))
        CF.fit_to_data=_orig; dist=cap["d"]
        x=jnp.hstack([jnp.asarray(Yv),jnp.asarray(uz,jnp.float32)]); cond=jnp.asarray(Xv); sl=True
        FB=TR[:cfg.batch_size]
        def gfn(pp,st,terms):
            gm=eqx.filter_grad(lambda q:-terms(q,st,x[FB],cond[FB])[1].mean())(pp)
            gc=eqx.filter_grad(lambda q:-terms(q,st,x[FB],cond[FB])[2].mean())(pp)
            ks=jax.tree_util.tree_leaves(jax.tree_util.tree_map_with_path(lambda a,_:jax.tree_util.keystr(a),pp))
            def vec(g): return np.concatenate([np.asarray(l).ravel() for k,l in zip(ks,jax.tree_util.tree_leaves(g)) if ".bijections[3]." in k])
            vm,vc=vec(gm),vec(gc); nm,nc=np.linalg.norm(vm),np.linalg.norm(vc)
            return dict(grad_margin_into_margin=float(nm),grad_copula_into_margin=float(nc),
                        grad_copula_into_margin_weighted=float(nc*1.0),
                        grad_cosine=float(vm@vc/max(nm*nc,1e-12)))
    print(f"  {npar(dist):,} trainable parameters",flush=True)
    def tfn(pp,st):
        y0,y1=sample_cond(pp,st,0,NMC_TRACK,Xv.shape[1],sl)
        m,_,_,_=evaluate(y0,y1,Y0.mean(0),Y1.mean(0),True,"track_"); return m
    bp,st,hist,info=run_loop(dist,x,cond,TR,VA,S["lr"],track_fn=tfn,grad_fn=gfn)
    y0,y1=sample_cond(bp,st,12345,NMC_FINAL,Xv.shape[1],sl)
    m,tau,e0,e1=evaluate(y0,y1,Y0.mean(0),Y1.mean(0),True,"final_")
else:
    tr0,va0=TR[Xv[TR,0]==0],VA[Xv[VA,0]==0]; tr1,va1=TR[Xv[TR,0]==1],VA[Xv[VA,0]==1]
    print(f"  untreated: {len(tr0)} train / {len(va0)} val   treated: {len(tr1)} train / {len(va1)} val",flush=True)
    d0=margin_dist(jr.PRNGKey(SEED_FIT),None); d1=margin_dist(jr.PRNGKey(SEED_FIT+1000),None)
    print(f"  {npar(d0):,} trainable parameters per flow",flush=True)
    bp0,st0,h0,i0=run_loop(d0,jnp.asarray(Yv),None,tr0,va0,S["lr"],label="_arm0",pfx="arm0_")
    bp1,st1,h1,i1=run_loop(d1,jnp.asarray(Yv),None,tr1,va1,S["lr"],label="_arm1",step0=len(h0),pfx="arm1_")
    hist=[{**a,**{f"arm1_{k}":v for k,v in b.items()}} for a,b in zip(h0,h1)]
    info={f"arm0_{k}":v for k,v in i0.items()}; info.update({f"arm1_{k}":v for k,v in i1.items()})
    info["termination"]=f"arm0={i0['termination']},arm1={i1['termination']}"
    info["converged"]=bool(i0["converged"] and i1["converged"])
    y0,y1=[],[]
    for b in range(0,NMC_SEP,2000):
        a,bb=sample_uncond(bp0,st0,bp1,st1,12345+b,min(2000,NMC_SEP-b)); y0.append(a); y1.append(bb)
    y0,y1=np.concatenate(y0),np.concatenate(y1)
    m,tau,e0,e1=evaluate(y0,y1,Y0.mean(0),Y1.mean(0),True,"final_")

oracle=min([h.get("track_ate_mae",np.inf) for h in hist]+[np.inf])
oe=[h["epoch"] for h in hist if h.get("track_ate_mae",np.inf)==oracle]
m.update({k:v for k,v in info.items() if isinstance(v,(int,float,bool,str))})
m["ORACLE_best_tracked_ate_mae"]=float(oracle) if np.isfinite(oracle) else None
m["ORACLE_best_tracked_epoch"]=oe[0] if oe else None
m["total_wall_s"]=time.monotonic()-t_all
np.savez(f"{OUT}/result.npz",tau_hat=tau,ATE=ate,e0_map=e0,e1_map=e1,train=TR,val=VA,
         **{k:np.array([h.get(k,np.nan) for h in hist]) for k in hist[0]})
json.dump({"metrics":m,"info":{k:(v if isinstance(v,(int,float,bool,str)) else list(v)) for k,v in info.items()},
           "config":dict(setup=setup,seed_fit=SEED_FIT,margin=MARGIN,lr=S["lr"])},open(f"{OUT}/result.json","w"),indent=1)
RUN.summary.update({k:v for k,v in m.items() if isinstance(v,(int,float,bool,str))})
print(f"\n##### {TAG}")
print(f"  termination: {info['termination']}   converged (patience): {info['converged']}   wall {m['total_wall_s']:.0f}s")
for k in ("final_ate_valid","final_nonfinite_total","final_ate_mae","final_reference_disc_signed_err",
          "final_reference_ring_signed_err","final_far_region_signed_err","final_reference_ring_se_paired",
          "final_reference_ring_e0","final_reference_ring_e1","final_y0_max_abs_finite","final_y1_max_abs_finite",
          "extreme_draw_influence_on_mae","ORACLE_best_tracked_ate_mae"):
    kk=k if k in m else "final_"+k
    if kk in m: print(f"  {k:<38} {m[kk]}")
RUN.finish()
