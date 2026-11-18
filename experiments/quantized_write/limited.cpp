// Fixed-cone normalized readout. Both paths share packing, step proposals and gradient code.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using Buffer = std::vector<double>;
constexpr int proposals = 8;
constexpr std::array<double, proposals> steps{0., .2, .1, .05, .025, .0125, .00625, .003125};

static double dot(const double* a, const double* b, int n) {
    return cblas_ddot(n, a, 1, b, 1);
}
static double norm(const double* a, int n) {
    return std::sqrt(dot(a,a,n));
}

// At each head the numerator of the normalized objective is linear in the
// three line-search coefficients. Its squared denominator is quadratic.
struct Moments {
    double aa=0., bb=0., cc=0., ab=0., ac=0., bc=0., ta=0., tb=0., tc=0.;
    double score(double a, double b, double c, int values) const {
        double squared = a*a*aa+b*b*bb+c*c*cc+2.*(a*b*ab+a*c*ac+b*c*bc);
        return (a*ta+b*tb+c*tc)/std::sqrt(std::max(0.,squared)/values+1e-6);
    }
};

extern "C" int solve_limited(int iterations, int use_moments, int heads, int keys, int values,
    const double* state, const double* delta, const double* native,
    const double* raw, double* output) {
    const int banks = 2*heads, columns = banks*values;
    const double qnorm = norm(native,keys), raw_norm = norm(raw,keys);
    bool nonzero = std::any_of(delta,delta+heads*keys*values,[](double x){return x!=0.;});
    if (!nonzero || std::min(qnorm,raw_norm)<1e-12) {
        std::copy(raw,raw+keys,output);
        return 0;
    }
    Buffer transform(keys*columns), n(keys), u(keys), gradient(keys), tangent(keys);
    Buffer base_read(columns), read(columns), tangent_read(columns), target(columns);
    Buffer derivative(columns), options(proposals*keys), option_reads(proposals*columns);
    for(int k=0;k<keys;++k) {
        n[k]=u[k]=native[k]/qnorm;
        for(int h=0;h<heads;++h) for(int v=0;v<values;++v) {
            int in=(h*keys+k)*values+v;
            transform[k*columns+h*values+v]=state[in]+delta[in];
            transform[k*columns+(h+heads)*values+v]=state[in];
        }
    }
    auto read_state=[&](const double* q, double scale, double* result){
        cblas_dgemv(CblasRowMajor,CblasTrans,keys,columns,scale,transform.data(),columns,
                    q,1,0.,result,1);
    };
    read_state(native,1.,base_read.data());
    read=base_read;
    for(int h=0;h<heads;++h) {
        int a=h*values, b=(h+heads)*values;
        double ra=std::sqrt(dot(&read[a],&read[a],values)/values+1e-6);
        double rb=std::sqrt(dot(&read[b],&read[b],values)/values+1e-6);
        for(int v=0;v<values;++v) target[a+v]=read[a+v]/ra-read[b+v]/rb;
    }
    double nt=std::max(norm(target.data(),heads*values),1e-30);
    for(int i=0;i<heads*values;++i){ target[i]/=nt; target[i+heads*values]=-target[i]; }
    for(int iter=0;iter<iterations;++iter) {
        for(int h=0;h<banks;++h) {
            int p=h*values;
            double var=dot(&read[p],&read[p],values)/values+1e-6;
            double proj=dot(&read[p],&target[p],values)/(values*var);
            for(int v=0;v<values;++v) derivative[p+v]=(target[p+v]-proj*read[p+v])/std::sqrt(var);
        }
        cblas_dgemv(CblasRowMajor,CblasNoTrans,keys,columns,qnorm,transform.data(),columns,
                    derivative.data(),1,0.,gradient.data(),1);
        double parallel=dot(gradient.data(),u.data(),keys);
        for(int k=0;k<keys;++k) tangent[k]=gradient[k]-u[k]*parallel;
        double tn=std::max(norm(tangent.data(),keys),1e-30);
        for(auto &x:tangent) x/=tn;
        std::array<double,proposals> a{},b{},c{},scores{};
        for(int j=0;j<proposals;++j) {
            auto* q=&options[j*keys];
            for(int k=0;k<keys;++k) q[k]=u[k]+steps[j]*tangent[k];
            double len=norm(q,keys);
            a[j]=1./len; b[j]=steps[j]/len;
            for(int k=0;k<keys;++k) q[k]/=len;
            double cosine=dot(q,n.data(),keys);
            if(cosine<.5) {
                for(int k=0;k<keys;++k) q[k]-=cosine*n[k];
                double factor=std::sqrt(.75)/std::max(norm(q,keys),1e-30);
                for(int k=0;k<keys;++k) q[k]=.5*n[k]+factor*q[k];
                a[j]*=factor; b[j]*=factor; c[j]=.5-factor*cosine;
            }
        }
        std::copy(u.begin(),u.end(),options.begin());
        a[0]=1.; b[0]=c[0]=0.;
        if(use_moments) {
            read_state(tangent.data(),qnorm,tangent_read.data());
            for(int h=0;h<banks;++h) {
                int p=h*values;
                Moments m;
                for(int v=0;v<values;++v) {
                    int i=p+v;
                    double x=read[i], y=tangent_read[i], z=base_read[i], t=target[i];
                    m.aa+=x*x; m.bb+=y*y; m.cc+=z*z;
                    m.ab+=x*y; m.ac+=x*z; m.bc+=y*z;
                    m.ta+=t*x; m.tb+=t*y; m.tc+=t*z;
                }
                for(int j=0;j<proposals;++j) scores[j]+=m.score(a[j],b[j],c[j],values);
            }
        } else {
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,proposals,columns,keys,
                        qnorm,options.data(),keys,transform.data(),columns,0.,option_reads.data(),columns);
            std::copy(read.begin(),read.end(),option_reads.begin());
            for(int j=0;j<proposals;++j) for(int h=0;h<banks;++h) {
                auto* x=&option_reads[j*columns+h*values];
                scores[j]+=dot(x,&target[h*values],values)/std::sqrt(dot(x,x,values)/values+1e-6);
            }
        }
        int best=int(std::max_element(scores.begin(),scores.end())-scores.begin());
        std::copy_n(&options[best*keys],keys,u.begin());
        if(use_moments) {
            for(int i=0;i<columns;++i) read[i]=a[best]*read[i]+b[best]*tangent_read[i]+c[best]*base_read[i];
        } else {
            std::copy_n(&option_reads[best*columns],columns,read.begin());
        }
    }
    double final_scale=raw_norm/norm(u.data(),keys);
    for(int k=0;k<keys;++k) output[k]=u[k]*final_scale;
    return 0;
}
