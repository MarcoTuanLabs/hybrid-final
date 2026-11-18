// Maximize the shared-head normalized read difference on a spherical cap.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using Vec=std::vector<double>;
static double dot(const Vec& a,const Vec& b){return cblas_ddot((int)a.size(),a.data(),1,b.data(),1);}
static double length(const Vec& a){return std::sqrt(dot(a,a));}
static void scale(Vec& a,double x){for(auto& v:a)v*=x;}
static void axpy(Vec& a,double x,const Vec& b){cblas_daxpy((int)a.size(),x,b.data(),1,a.data(),1);}

struct Moment {
    double rr,rw,rn,ww,wn,nn,tr,tw,tn;
    double value(double a,double b,double c,double d) const {
        double squared=a*a*rr+b*b*ww+c*c*nn+2*(a*b*rw+a*c*rn+b*c*wn);
        return (a*tr+b*tw+c*tn)/std::sqrt(std::max(0.,squared)/d+1e-6);
    }
};
struct ArcPoint {double a,b,c;};

// Modes: 0 projected steepest, 1 projected CG, 2 active steepest,
// 3 active CG, 4 active CG with a fixed seven-step comparison grid.
extern "C" int arc_solve(int mode,int steps,int heads,int k,int d,
    const double* state,const double* delta,const double* native,const double* raw,double* output,
    double* info){
    constexpr double cap=.5, side=.86602540378443864676;
    const int h2=2*heads,cols=h2*d;
    Vec n(native,native+k),orig(raw,raw+k);
    double qnorm=length(n),rawnorm=length(orig);
    if(qnorm<1e-12 || rawnorm<1e-12 ||
       std::none_of(delta,delta+heads*k*d,[](double x){return x!=0.;})){
        std::copy(raw,raw+k,output);info[0]=info[1]=0;return 0;
    }
    scale(n,1/qnorm);Vec u=n,g(k),prev_g(k),prev_dir(k),dir(k),w(k),boundary(k);
    Vec A(k*cols),base(cols),r(cols),wr(cols),t(cols),jac(cols);
    for(int i=0;i<k;i++)for(int h=0;h<heads;h++)for(int j=0;j<d;j++){
        int v=(h*k+i)*d+j;
        A[i*cols+h*d+j]=state[v]+delta[v];
        A[i*cols+(h+heads)*d+j]=state[v];
    }
    auto read=[&](const Vec& q, Vec& result){
        cblas_dgemv(CblasRowMajor,CblasTrans,k,cols,qnorm,A.data(),cols,q.data(),1,0.,result.data(),1);
    };
    auto inner=[&](const Vec& x,const Vec& y,int h){
        return cblas_ddot(d,x.data()+h*d,1,y.data()+h*d,1);
    };
    read(u,base);r=base;
    double tz=0.;
    for(int h=0;h<heads;h++){
        double re=std::sqrt(inner(r,r,h)/d+1e-6),rb=std::sqrt(inner(r,r,h+heads)/d+1e-6);
        for(int j=0;j<d;j++){
            int a=h*d+j,b=(h+heads)*d+j;
            t[a]=r[a]/re-r[b]/rb;t[b]=-t[a];tz+=t[a]*t[a];
        }
    }
    scale(t,1/std::max(std::sqrt(tz),1e-30));
    std::vector<Moment> moments(h2);
    for(int h=0;h<h2;h++){moments[h].nn=inner(base,base,h);moments[h].tn=inner(t,base,h);}
    bool cg=mode==1 || mode==3 || mode==4;
    int last_surface=-1,active_steps=0,accepted=0;
    for(int it=0;it<steps;it++){
        for(int h=0;h<h2;h++){
            double var=inner(r,r,h)/d+1e-6,ratio=inner(r,t,h)/(d*var);
            for(int j=h*d;j<(h+1)*d;j++)jac[j]=(t[j]-r[j]*ratio)/std::sqrt(var);
        }
        cblas_dgemv(CblasRowMajor,CblasNoTrans,k,cols,qnorm,A.data(),cols,jac.data(),1,0.,g.data(),1);
        axpy(g,-dot(g,u),u);
        double nu=std::clamp(dot(n,u),-1.,1.);
        bool surface=mode>=2 && nu<=cap+1e-9 && dot(g,n)<0;
        if(surface){
            boundary=n;axpy(boundary,-nu,u);
            scale(boundary,1/std::max(length(boundary),1e-30));
            axpy(g,-dot(g,boundary),boundary);
            active_steps++;
        }
        dir=g;
        if(cg && last_surface==(int)surface){
            Vec pg=prev_g, pd=prev_dir;
            axpy(pg,-dot(pg,u),u);axpy(pd,-dot(pd,u),u);
            if(surface){axpy(pg,-dot(pg,boundary),boundary);axpy(pd,-dot(pd,boundary),boundary);}
            double beta=std::max(0.,(dot(g,g)-dot(g,pg))/std::max(dot(prev_g,prev_g),1e-30));
            axpy(dir,beta,pd);
            if(dot(dir,g)<=1e-12*dot(g,g))dir=g;
        }
        if(length(dir)<1e-15)break;
        prev_g=g;prev_dir=dir;last_surface=surface;
        w=dir;scale(w,1/length(w));
        read(w,wr);
        double nw=dot(n,w);
        for(int h=0;h<h2;h++){
            auto& m=moments[h];
            m.rr=inner(r,r,h);m.rw=inner(r,wr,h);m.rn=inner(r,base,h);
            m.ww=inner(wr,wr,h);m.wn=inner(wr,base,h);m.tr=inner(t,r,h);m.tw=inner(t,wr,h);
        }
        auto point=[&](double angle){
            double a=std::cos(angle),b=std::sin(angle);
            if(surface)return ArcPoint{a,side*b,cap*(1-a)};
            double nc=a*nu+b*nw;
            if(nc>=cap)return ArcPoint{a,b,0.};
            double f=side/std::sqrt(std::max(1.-nc*nc,1e-16));
            return ArcPoint{f*a,f*b,cap-f*nc};
        };
        auto value=[&](double angle){
            ArcPoint p=point(angle);double total=0;
            for(auto& m:moments)total+=m.value(p.a,p.b,p.c,d);
            return total;
        };
        double best=0.,val=value(0.);
        auto test=[&](double ang){
            if(ang<0 || ang>1.6)return;
            double f=value(ang);
            if(f>val){val=f;best=ang;}
        };
        if(mode==4){
            for(double x:{.2,.1,.05,.025,.0125,.00625,.003125})test(std::atan(x));
        }else{
            for(double x:{.0125,.025,.05,.1,.2,.4,.8,1.2,1.6})test(x);
            double step=best<=.4?std::max(.00625,best/2):.2;
            for(int pass=0;pass<4;pass++){
                double center=best;
                test(center-step);test(center+step);step*=.5;
            }
        }
        if(best==0.)break;
        accepted++;
        auto p=point(best);
        for(int i=0;i<k;i++)u[i]=p.a*u[i]+p.b*w[i]+p.c*n[i];
        for(int i=0;i<cols;i++)r[i]=p.a*r[i]+p.b*wr[i]+p.c*base[i];
    }
    scale(u,rawnorm/length(u));std::copy(u.begin(),u.end(),output);
    info[0]=active_steps;info[1]=accepted;
    return 0;
}
