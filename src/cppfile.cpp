// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <random>
//#include <chrono>
#include <vector>
#include <limits>
#include <algorithm>    // std::max

// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> MatrixXl;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> VectorXl;
typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> VectorXs;

// [[Rcpp::export]]
Eigen::VectorXd llinsolve(const Eigen::MatrixXd & A, const Eigen::VectorXd & b){
  MatrixXl aa = A.cast<long double>();
  VectorXl bb = b.cast<long double>();
  VectorXl xx = aa.colPivHouseholderQr().solve(bb);
  Eigen::VectorXd x = xx.cast<double>();
  return x;
} 

Eigen::VectorXd solve(const Eigen::MatrixXd & A, const Eigen::VectorXd & b){
  return A.colPivHouseholderQr().solve(b);
} 

// [[Rcpp::export]]
Eigen::MatrixXd tsolveAndMultiply(const Eigen::MatrixXd & A, const Eigen::MatrixXd & C){
  Eigen::MatrixXd M = A.triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(C);
  return M; // C %*% solve(A)
} 

// [[Rcpp::export]]
Rcpp::List nullSpace(const Eigen::MatrixXd M){ // avec QR: voir MASS::Null
  Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
  Eigen::MatrixXd nspace = lu.kernel(); // not orthonormal
  int r = lu.rank();
  return Rcpp::List::create(Rcpp::Named("kernel") = nspace,
                            Rcpp::Named("rank") = r);
}

// [[Rcpp::export]]
Eigen::MatrixXd kernel(const Eigen::MatrixXd & M){ // ?
  Eigen::MatrixXd Mt = M.transpose();
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr1(Mt);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr2 = Mt.householderQr();
  Eigen::MatrixXd Q = qr2.householderQ();
  int r = qr1.rank();
  return Q.rightCols(Q.cols()-r); 
}
    
// [[Rcpp::export]]
Rcpp::List QRdecomp(const Eigen::MatrixXd & M){ // for nrows >= ncols
  Eigen::HouseholderQR<Eigen::MatrixXd> qr = M.householderQr();
  Eigen::MatrixXd R_ = qr.matrixQR().triangularView<Eigen::Upper>();
  Eigen::MatrixXd Q_ = qr.householderQ();
  Eigen::MatrixXd R = R_.block(0,0,M.cols(),M.cols());
  Eigen::MatrixXd Q = Q_.block(0,0,M.rows(),M.cols());
  return Rcpp::List::create(Rcpp::Named("Q") = Q,
                            Rcpp::Named("R") = R);
}

int rankMatrix(const Eigen::MatrixXd & M){
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(M);
  return qr.rank();
}
  
// [[Rcpp::export]]
Eigen::VectorXd whichLU(const Eigen::VectorXd & V, double L, double U){
  std::vector<size_t> whichl, checkl, whichu, checku, both;
  for(size_t i=0; i<V.size(); i++){
    bool b = false;
    if(V(i) >= L){
      whichl.push_back(i);
      b = true;
    }else{
      checkl.push_back(i);
    }
    if(V(i) <= U){
      whichu.push_back(i);
      if(b){
        both.push_back(i);
      }
    }else{
      checku.push_back(i);
    }
  }
  Eigen::VectorXd W(whichl.size());
  for(size_t i=0; i < whichl.size(); i++){
    W(i) = V(whichl[i]);
  }
  return W;
} // pour fidVertex, retourner une struct, ou bien faire direct dans fidVertex

// [[Rcpp::export]]
int main(){
  Eigen::VectorXd V(8);
  V << 0.0, 1.0, 5.0, 4.0, 9.0, 1.0, 2.0, 3.0;
  Eigen::VectorXd W = whichLU(V, 3.0, 10000.0);
  std::cout << W << std::endl;
  return 0;
}

//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator; // (seed);
//generator.seed(666);
// std::random_device rd;
// std::mt19937 generator(rd());
// std::mt19937 generator;

std::normal_distribution<double> distribution(0.0,1.0);

Eigen::MatrixXd gmatrix(size_t nrows, size_t ncols){
  Eigen::MatrixXd G(nrows,ncols);
  for(size_t i=0; i<nrows; i++){
    for(size_t j=0; j<ncols; j++){
      G(i,j) = distribution(generator);
    }
  }
  return G;
}


// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> ListOfGmatrices(
    size_t nrows, size_t ncols, size_t nmatrices){
  std::vector<Eigen::MatrixXd> matrices(nmatrices);
  for(unsigned i=0; i<nmatrices; i++){
    matrices[i] = gmatrix(nrows,ncols);
  }
  matrices[1](0,0) = 99;
  // Eigen::MatrixXd M = matrices[1];
  // M(0,0) = 99;
  // matrices[1] = M;
  for(unsigned i=0; i<nmatrices; i++){
    std::cout << matrices[i] << std::endl;
  }
  return matrices;
}


// [[Rcpp::export]]
Eigen::MatrixXd testcbind(){
  Eigen::MatrixXd M(3,0);
  Eigen::MatrixXd A = gmatrix(3,1);
  //Eigen::MatrixXd B(3,1);
  //B << M,A;
  //return B;
  Eigen::MatrixXd B;
  B = M;
  M.conservativeResize(Eigen::NoChange, 1);
  M << B,A;
  B = M;
  M.conservativeResize(Eigen::NoChange, 2);
  M << B,A;
  return M;
}

// [[Rcpp::export]]
int temp(unsigned n){
  int x = 0;
  for(unsigned i=0; i<n; i++){
    x = x+1;
  }
  return x;
}

// [[Rcpp::export]]
Rcpp::List fidVertex(Eigen::MatrixXd VT1, Eigen::MatrixXi CC1, 
                          Eigen::VectorXd VTsum, double L, double U, 
                          size_t Dim, int n, int k){
  size_t p = VTsum.size();
  std::vector<size_t> whichl, checkl, whichu, checku, both;
  for(size_t i=0; i<(size_t)VTsum.size(); i++){
    bool b = false;
    if(VTsum(i) >= L){
      whichl.push_back(i);
      b = true;
    }else{
      checkl.push_back(i);
    }
    if(VTsum(i) <= U){
      whichu.push_back(i);
      if(b){
        both.push_back(i);
      }
    }else{
      checku.push_back(i);
    }
  }
  Eigen::MatrixXi CCtemp(Dim, 0);
  Eigen::MatrixXd VTtemp(Dim, 0);
  int vert = 0;

  size_t lcheckl = checkl.size();
  size_t lwhichl = whichl.size();
  if(lcheckl < p){
    Eigen::MatrixXi CA(Dim, lcheckl); 
    Eigen::MatrixXi CB(Dim, lwhichl);
    for(size_t i=0; i < Dim; i++){
      for(size_t j=0; j < lcheckl; j++){
        CA(i,j) = CC1(i, checkl[j]);
      }
      for(size_t j=0; j < lwhichl; j++){
        CB(i,j) = CC1(i, whichl[j]);
      }
    }
    Eigen::MatrixXi INT = Eigen::MatrixXi::Zero(2*n,lcheckl);
    for(size_t ll=0; ll<lcheckl; ll++){
      for(size_t i=0; i<Dim; i++){
        INT(CA(i,ll),ll) = 1;
      }
    }
    Eigen::VectorXd VTsum_cl(lcheckl);
    Eigen::VectorXd VTsum_wl(lwhichl);
    Eigen::MatrixXd VT1_cl(Dim, lcheckl);
    Eigen::MatrixXd VT1_wl(Dim, lwhichl);
    for(size_t i=0; i<lcheckl; i++){
      VTsum_cl(i) = VTsum(checkl[i]);
      for(size_t j=0; j<Dim; j++){
        VT1_cl(j,i) = VT1(j,checkl[i]);
      }
    }
    for(size_t i=0; i<lwhichl; i++){
      VTsum_wl(i) = VTsum(whichl[i]);
      for(size_t j=0; j<Dim; j++){
        VT1_wl(j,i) = VT1(j,whichl[i]);
      }
    }
    for(size_t ii=0; ii<p-lcheckl; ii++){
      Eigen::MatrixXi INT2(Dim, lcheckl);
      for(size_t i=0; i<Dim; i++){
        for(size_t j=0; j<lcheckl; j++){
          INT2(i,j) = INT(CB(i,ii),j);
        }
      }
      for(size_t j=0; j<lcheckl; j++){
        int colSum = 0;
        for(size_t i=0; i<Dim; i++){
          colSum += INT2(i,j);
        }
        if(colSum == (int)Dim-1){
          vert += 1;
          // std::vector<int> which1;
          // for(size_t i=0; i<Dim; i++){
          //   if(INT2(i,j)==1){
          //     which1.push_back(CB(i,ii));
          //   }
          // }
          std::vector<int> inter(Dim);
          size_t m = 0;
          for(size_t i=0; i<Dim; i++){
            if(INT2(i,j)==1){
              inter[m] = CB(i,ii);
              m += 1;
            }
          }
          inter[Dim-1] = k+n;
          // Eigen::VectorXi inter(which1.size()+1);
          // for(size_t i=0; i<which1.size(); i++){
          //   inter(i) = which1[i]; // ? which1.size() = Dim-1 ?
          // }
          // inter(which1.size()) = k+n;
          // Eigen::MatrixXi M;
          // M = CCtemp;
          CCtemp.conservativeResize(Eigen::NoChange, vert); 
          // CCtemp << M,inter; // rq: on pourrait seulement append la dernière colonne
          for(size_t i=0; i<Dim; i++){
            CCtemp(i,vert-1) = inter[i]; 
          }
          double lambda = (L-VTsum_wl(ii))/(VTsum_cl(j)-VTsum_wl(ii));
          // Eigen::VectorXd vtnew(Dim);
          // for(size_t i=0; i<Dim; i++){
          //   vtnew(i) = lambda*VT1_cl(i,j) + (1-lambda)*VT1_wl(i,ii);
          // }
          // Eigen::MatrixXd MM;
          // MM = VTtemp;
          // VTtemp.conservativeResize(Eigen::NoChange, VTtemp.cols()+1);
          // VTtemp << MM,vtnew;
          VTtemp.conservativeResize(Eigen::NoChange, vert);
          for(size_t i=0; i<Dim; i++){
            VTtemp(i,vert-1) = lambda*VT1_cl(i,j) + (1-lambda)*VT1_wl(i,ii);
          }
        }
      }
    }
  }
  
  size_t lchecku = checku.size();
  size_t lwhichu = whichu.size();
  if(lchecku < p){
    Eigen::MatrixXi CA(Dim, lchecku); 
    Eigen::MatrixXi CB(Dim, lwhichu);
    for(size_t i=0; i < Dim; i++){
      for(size_t j=0; j < lchecku; j++){
        CA(i,j) = CC1(i, checku[j]);
      }
      for(size_t j=0; j < lwhichu; j++){
        CB(i,j) = CC1(i, whichu[j]);
      }
    }
    Eigen::MatrixXi INT = Eigen::MatrixXi::Zero(2*n,lchecku);
    for(size_t ll=0; ll<lchecku; ll++){
      for(size_t i=0; i<Dim; i++){
        INT(CA(i,ll),ll) = 1;
      }
    }
    Eigen::VectorXd VTsum_cu(lchecku);
    Eigen::VectorXd VTsum_wu(lwhichu);
    Eigen::MatrixXd VT1_cu(Dim, lchecku);
    Eigen::MatrixXd VT1_wu(Dim, lwhichu);
    for(size_t i=0; i<lchecku; i++){
      VTsum_cu(i) = VTsum(checku[i]);
      for(size_t j=0; j<Dim; j++){
        VT1_cu(j,i) = VT1(j,checku[i]);
      }
    }
    for(size_t i=0; i<lwhichu; i++){
      VTsum_wu(i) = VTsum(whichu[i]);
      for(size_t j=0; j<Dim; j++){
        VT1_wu(j,i) = VT1(j,whichu[i]);
      }
    }
    for(size_t ii=0; ii<p-lchecku; ii++){
      Eigen::MatrixXi INT2(Dim, lchecku);
      for(size_t i=0; i<Dim; i++){
        for(size_t j=0; j<lchecku; j++){
          INT2(i,j) = INT(CB(i,ii),j);
        }
      }
      for(size_t j=0; j<lchecku; j++){
        int colSum = 0;
        for(size_t i=0; i<Dim; i++){
          colSum += INT2(i,j);
        }
        if(colSum == (int)Dim-1){
          vert += 1;
          std::vector<int> inter(Dim);
          size_t m = 0;
          for(size_t i=0; i<Dim; i++){
            if(INT2(i,j)==1){
              inter[m] = CB(i,ii);
              m += 1;
            }
          }
          inter[Dim-1] = k;
          CCtemp.conservativeResize(Eigen::NoChange, vert); 
          for(size_t i=0; i<Dim; i++){
            CCtemp(i,vert-1) = inter[i]; 
          }
          double lambda = (U-VTsum_wu(ii))/(VTsum_cu(j)-VTsum_wu(ii));
          VTtemp.conservativeResize(Eigen::NoChange, vert);
          for(size_t i=0; i<Dim; i++){
            VTtemp(i,vert-1) = lambda*VT1_cu(i,j) + (1-lambda)*VT1_wu(i,ii);
          }
        }
      }
    }
  }
  
  size_t lboth = both.size();
  if(lboth>0){
    for(size_t j=0; j<lboth; j++){
      vert += 1;
      CCtemp.conservativeResize(Eigen::NoChange, vert); 
      VTtemp.conservativeResize(Eigen::NoChange, vert); 
      for(size_t i=0; i<Dim; i++){
        CCtemp(i,vert-1) = CC1(i,both[j]);
        VTtemp(i,vert-1) = VT1(i,both[j]);
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("CCtemp") = CCtemp,
                            Rcpp::Named("VTtemp") = VTtemp,
                            Rcpp::Named("vert") = vert);
}


int sgn(double x){
  return (0 < x) - (x < 0);
}

//const long double PI = 3.141592653589793238L;
// const double PI = 3.141592653589793; already defined

std::uniform_real_distribution<double> runif(0.0,1.0);

double approx(double x, unsigned n){
  return round(x * pow(10, n)) / pow(10, n);
}

// [[Rcpp::export]]
Rcpp::List fidSample(Eigen::VectorXd VT2, Eigen::VectorXd VTsum, 
                     double L, double U){
  double ZZ, wt; // outputs
  size_t p = VTsum.size(); // = VT2.size()
  std::vector<size_t> high, low, zero, zeronot;
  size_t lhigh=0, llow=0, lzero=0, lzeronot;
  for(size_t i=0; i<p; i++){
    if(VT2(i)==0){
      lzero += 1;
      zero.push_back(i);
    }else{
      zeronot.push_back(i);
      if(VT2(i)>0){
        lhigh += 1;
        high.push_back(i);
      }else{
        llow += 1;
        low.push_back(i);
      }
    }
  }
  lzeronot = p-lzero;
  double MAX, MIN;
  double infty = std::numeric_limits<double>::infinity();
  if((lhigh>0 && llow>0) || lzero>0){
    double temp;
    std::vector<int> UU(p);
    std::vector<int> LL(p);
    std::vector<int> SS(p);
    for(size_t i=0; i<p; i++){
      UU[i] = sgn(U-VTsum(i));
      LL[i] = sgn(L-VTsum(i));
      SS[i] = sgn(VT2(i));
    }
    if(lzero==p){
      MAX = infty;
      MIN = -infty;
      bool anyUUpos = false, anyLLneg = false;
      size_t i = 0;
      while(i<p && !anyUUpos){
        if(UU[i]>0){
          anyUUpos = true;
        }
        i += 1;
      }
      i = 0;
      while(i<p && !anyLLneg){
        if(LL[i]<0){
          anyLLneg = true;
        }
        i += 1;
      }
      temp = anyUUpos && anyLLneg;
    }else{
      size_t i = 0;
      bool c1 = true, c2 = true, d1 = true, d2 = true;
      while(i < lzero && c1){
        if(UU[zero[i]]==-1 || LL[zero[i]]==-1){
          c1 = false;
        }
        i += 1;
      }
      i = 0;
      while(i < lzero && c2){
        if(UU[zero[i]]==1 || LL[zero[i]]==1){
          c2 = false;
        }
        i += 1;
      }
      i = 0;
      while(i < p && d1){
        if(SS[i] == -1){
          d1 = false;
        }
        i += 1;
      }
      i = 0;
      while(i < p && d2){
        if(SS[i] == 1){
          d2 = false;
        }
        i += 1;
      }
      if((d1 && c1) || (d2 && c2)){
        MAX = infty;
        MIN = infty;
        for(size_t i=0; i<lzeronot; i++){
          size_t zni = zeronot[i];
          MIN = std::min(MIN, std::min((U-VTsum(zni))/VT2(zni),(L-VTsum(zni))/VT2(zni)));
        }
        temp = 1-(atan(MIN)/PI+0.5);
      }else if((d2 && c1) || (d1 && c2)){
        MIN = -infty;
        MAX = -infty;
        for(size_t i=0; i<lzeronot; i++){
          size_t zni = zeronot[i];
          MAX = std::max(MAX, std::max((U-VTsum(zni))/VT2(zni),(L-VTsum(zni))/VT2(zni)));
        }
        temp = atan(MAX)/PI+0.5; 				
      }else{
        double Hmax = -infty;
        double Hmin = infty;
        for(size_t i=0; i<lhigh; i++){
          size_t hi = high[i];
          double xu = (U-VTsum(hi))/VT2(hi);
          double xl = (L-VTsum(hi))/VT2(hi);
          Hmax = std::max(Hmax, std::max(xu,xl));
          Hmin = std::min(Hmin, std::min(xu,xl));
        }
        double Lmax = -infty;
        double Lmin = infty;
        for(size_t i=0; i<llow; i++){
          size_t li = low[i];
          double xu = (U-VTsum(li))/VT2(li);
          double xl = (L-VTsum(li))/VT2(li);
          Lmax = std::max(Lmax, std::max(xu,xl));
          Lmin = std::min(Lmin, std::min(xu,xl));
        }
        double bpos, tpos, bneg, tneg;
        if(approx(Lmin-Hmax,12)>=0){
          bpos = -infty;
          tpos = Hmax;
          bneg = Lmin;
          tneg = infty;
        }else if(approx(Hmin-Lmax,12)>=0){
          bpos = Hmin;
          tpos = infty;
          bneg = -infty;
          tneg = Lmax;
        }else{
          bpos = -infty;
          tpos = infty;
          bneg = -infty;
          tneg = infty;
        }
        double Pprob, Nprob;
        if(tpos==infty){
          Pprob = 1-(atan(bpos)/PI+0.5);
        }else{
          Pprob = atan(tpos)/PI+0.5;
        }
        if(tneg==infty){
          Nprob = 1-(atan(bneg)/PI+0.5);
        }else{
          Nprob = atan(tneg)/PI+0.5;
        }
        temp = Pprob+Nprob;
        Pprob = Pprob/temp;
        Nprob = 1-Pprob;
        if(runif(generator) <= Nprob){
          MIN = bneg;
          MAX = tneg;
        }else{
          MIN = bpos;
          MAX = tpos;
        }
      }
    }
    double y = atan(MAX)/PI+0.5;
    double x = atan(MIN)/PI+0.5;
    double u = x+(y-x)*runif(generator);
    ZZ = tan(PI*(u-0.5));
    double ZZ2 = ZZ*ZZ;
    wt = exp(-ZZ2/2)*(1+ZZ2)*temp;
  }else{
    MAX = -infty;
    MIN = infty;
    for(size_t i=0; i<p; i++){
      double xu = (U-VTsum(i))/VT2(i);
      double xl = (L-VTsum(i))/VT2(i);
      MAX = std::max(MAX, std::max(xu,xl));
      MIN = std::min(MIN, std::min(xu,xl));
    }
    double y = atan(MAX)/PI+0.5; 
    double x = atan(MIN)/PI+0.5;
    double u = x + (y-x)*runif(generator);
    ZZ = tan(PI*(u-0.5));
    double ZZ2 = ZZ*ZZ;
    wt = exp(-ZZ2/2)*(1+ZZ2)*(y-x);
  }
  
  return Rcpp::List::create(Rcpp::Named("ZZ") = ZZ,
                            Rcpp::Named("wt") = wt);
}

// [[Rcpp::export]]
Eigen::VectorXi cppunique(Eigen::VectorXi v){
  int size = v.size();
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size;) {
      if (v(j) == v(i)) {
        for (int k = j; k+1 < size; k++) {
          v(k) = v(k + 1);
        }
        size--;
      } else {
        j++;
      }
    }
  }
  Eigen::VectorXi out = v.head(size);
  return out;
} 

std::vector<std::vector<int>> cartesianProduct(const std::vector<std::vector<int>>& v){
  std::vector<std::vector<int>> s = {{}};
  for(auto& u : v){
    std::vector<std::vector<int>> r;
    for(auto& x : s){
      for(auto y : u){
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s.swap(r);
  }
  return s;
}

std::vector<std::vector<int>> combinations(std::vector<int> C, int n){
  std::vector<std::vector<int>> sets;
  for(size_t i=0; i<C.size(); i++){
    sets.push_back({C[i],C[i]+n});
  }
  return cartesianProduct(sets);
}

Eigen::MatrixXi vv2matrix(std::vector<std::vector<int>> U, size_t nrow, size_t ncol){
  Eigen::MatrixXi out(nrow,ncol);
  for(size_t i=0; i<nrow; i++){
    for(size_t j=0; j<ncol; j++){
      out(i,j) = U[j][i];
    }
  }
  return out;
}

size_t spow(size_t base, size_t exp){
  size_t result = 1;
  while(exp){
    if(exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

// [[Rcpp::export]]
Eigen::VectorXd Vsort(Eigen::VectorXd V){
  std::sort(V.data(),V.data()+V.size());
  return V;
}

double Vproduct(Eigen::VectorXd vec){
  double prod = 1;
  for (int i=0; i<vec.size(); i++) {
    prod *= vec(i);
  }
  return prod;
}

double Vsum(Eigen::VectorXd vec){
  double sum = 0;
  for (int i=0; i<vec.size(); i++) {
    sum += vec(i);
  }
  return sum;
}

std::vector<double> Vcumsum(Eigen::VectorXd vec){
  std::vector<double> out(vec.size());
  out[0] = vec(0);
  for(int i=1; i<vec.size(); i++) {
    out[i] = out[i-1] + vec(i);
  }
  return out;
}

// std::random_device rd;
// std::mt19937 g(rd());

template <class T>
std::vector<T> zero2n(T n){
  std::vector<T> out(n);
  for(T i=0; i<n; i++){
    out[i] = i;
  }
  return out;
}

// [[Rcpp::export]]
std::vector<size_t> sample_int(size_t n){
  std::vector<size_t> elems = zero2n(n);
  std::shuffle(elems.begin(), elems.end(), generator);
  return elems;
}

// [[Rcpp::export]]
Rcpp::List gfimm_(Eigen::VectorXd L, Eigen::VectorXd U, 
                  Eigen::MatrixXd FE, Eigen::MatrixXd RE, 
                  Eigen::MatrixXi RE2, Rcpp::IntegerVector E,
                  size_t N, size_t thresh){
  Eigen::VectorXd WTnorm(N); // output:weights
  const size_t n = L.size();
  const size_t fe = FE.cols(); // si FE=NULL, passer une matrice n x 0
  const size_t re = RE2.cols();
  const size_t Dim = fe+re;
  const Rcpp::IntegerVector Esum = Rcpp::cumsum(E);
  
  //-------- SET-UP ALGORITHM OBJECTS ------------------------------------------
  std::vector<Eigen::MatrixXd> Z(re);
  std::vector<Eigen::MatrixXd> weight(re);
  Rcpp::NumericVector ESS(n, (double)N);
  //std::vector<size_t> VC(N); // Number of vertices
  std::vector<Eigen::MatrixXi> CC(N); // constraints 
  std::vector<int> C; // initial constraints 
  std::vector<int> K; // complement of C
  std::vector<Eigen::MatrixXd> VT(N); // vertices
  std::vector<Eigen::MatrixXd> VT0(N); // vertices
  
  //-------- SAMPLE ALL Z's / SET-UP WEIGHTS -----------------------------------
  std::vector<Eigen::MatrixXd> A(N);
  for(size_t j=0; j<re; j++){  
    Z[j] = gmatrix(E(j),N); 
    weight[j] = Eigen::MatrixXd::Ones(E(j),N);
    Eigen::MatrixXd M = RE.block(0, Esum(j)-E(j), n, E(j)) * Z[j];
    for(size_t k=0; k<N; k++){
      A[k].resize(n,re);
      for(size_t i=0; i<n; i++){
        A[k](i,j) = M(i,k);
      }
    } 
  }      
  
  Eigen::MatrixXd AA(n, Dim);
  AA << FE,A[1];
  Eigen::MatrixXd AT(0, Dim);
  //Eigen::MatrixXd Atemp(0, Dim);
  int r = 0;
  for(size_t i=0; i<n; i++){
    Eigen::MatrixXd Atemp(AT.rows()+1,Dim);
    Atemp << AT, AA.row(i);
    if(rankMatrix(Atemp) > r){
      AT.conservativeResize(AT.rows()+1,Eigen::NoChange);
      for(size_t j=0; j<Dim; j++){
        AT(AT.rows()-1,j) = AA(i,j);
      }
      r = rankMatrix(AT);
      C.push_back((int)i);
    }else{
      K.push_back((int)i);
    }
  }

  std::vector<int> K_start = C;
  Eigen::VectorXi CVec = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(C.data(), Dim);
  for(size_t i=0; i<n-Dim; i++){
    for(size_t j=0; j<N; j++){
      Z[re-1](K[i],j) = 0;
    }
  }
  
  //-------- FIND INITIAL VERTICES ---------------------------------------------
  std::vector<std::vector<int>> USE = combinations(C,n);
  size_t twoPowerDim = spow(2, Dim);
  Eigen::MatrixXi tUSE = vv2matrix(USE, Dim, twoPowerDim);
  // me semble pas nécessaire de faire CC <- rep(list(t(USE)))
  Eigen::VectorXd b(2*n);
  b << U, -L;
  Eigen::MatrixXd FEFE(2*n,fe);
  FEFE << FE,-FE;
  for(size_t k=0; k<N; k++){
    Eigen::MatrixXd V(Dim, twoPowerDim);
    Eigen::MatrixXd AkAk(2*n,re);
    AkAk << A[k], -A[k];
    Eigen::MatrixXd AA(2*n,Dim);
    AA << FEFE,AkAk;
    for(size_t i=0; i<twoPowerDim; i++){
      Eigen::MatrixXd AAuse(Dim,Dim);
      Eigen::VectorXd buse(Dim);
      for(size_t j=0; j<Dim; j++){
        buse(j) = b(USE[i][j]);
        for(size_t l=0; l<Dim; l++){
          AAuse(j,l) = AA(USE[i][j],l);
        }
      }  
      V.col(i) = solve(AAuse, buse);
    }
    VT0[k] = V;
  }
  std::vector<size_t> VC(N, twoPowerDim);

  //-------- MAIN ALGORITHM ----------------------------------------------------
  //double break_point = 10;
  double lengthK = (double)(n-Dim);
  size_t K_n = (size_t)(ceil(lengthK/10.0));
  std::vector<Eigen::VectorXi> K_temp(K_n);
  Eigen::VectorXi KV = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(K.data(), n-Dim);
  for(size_t i=0; i<K_n-1; i++){
    K_temp[i] = KV.segment(i*10, 10);
  }
  K_temp[K_n-1] = KV.tail(n-Dim-(K_n-1)*10);
  Eigen::VectorXi K1(0);
  for(size_t k_n=0; k_n<K_n; k_n++){
    K1.conservativeResize(K1.size()+K_temp[k_n].size());
    K1.tail(K_temp[k_n].size()) = K_temp[k_n];
    for(int ki=0; ki<K_temp[k_n].size(); ki++){
      int k = K_temp[k_n](ki);
      if(k_n>0){
        for(size_t i=0; i<re; i++){
          if(E[i]>Z[i].rows()){ // je n'ai pas trouvé où Z[i].rows change
            int nrowsZi = Z[i].rows();
            Z[i].conservativeResize(E(i), Eigen::NoChange);
            Z[i].bottomRows(E(i)-nrowsZi) = gmatrix(E(i)-nrowsZi, N);
          }
        }
      }
      for(size_t i=0; i<N; i++){
        Eigen::MatrixXd VTi = VT0[i];
        Eigen::MatrixXd VT1 = VTi.topRows(Dim-1);
        Eigen::VectorXd VT2 = VTi.row(Dim-1);
        Eigen::VectorXd Z1t(re-1);
        for(size_t j=0; j<re-1; j++){
          Z1t(j) = Z[j](RE2(k,j),i);
        }
        Eigen::VectorXd Z1(Dim-1);
        Z1 << FE.row(k), Z1t;
        Eigen::VectorXd VTsum = VT1.transpose() * Z1;
        Rcpp::List sample = fidSample(VT2, VTsum, L(k), U(k));
        double ZZ = sample["ZZ"];
        double wt = sample["wt"];
        Z[re-1](k,i) = ZZ;
        weight[re-1](k,i) = wt;
        VTsum += ZZ * VT2;
        Rcpp::List vertex = fidVertex(VTi, tUSE, VTsum, L(k), U(k), Dim, (int)n, k);
        VC[i] = vertex["vert"];
        // CC[i].resize(Dim, VC(i));
        CC[i] = vertex["CCtemp"];
        // VT[i].resize(Dim, VC(i));
        VT[i] = vertex["VTtemp"];
      }
      Eigen::VectorXd WT(N);
      for(size_t j=0; j<N; j++){
        WT(j) = Vproduct(weight[re-1].col(j));
      }
      double WTsum = Vsum(WT);
      WTnorm = WT/WTsum;
      ESS(k) = 1/WTnorm.dot(WTnorm);
      if(ESS(k)<thresh && k < K[n-Dim-1]){
        std::vector<size_t> N_sons(N,0);
        std::vector<double> dist = Vcumsum(WTnorm);
        double aux = runif(generator);
        
        Rcpp::Rcout << "aux " << aux << std::endl;
        
        std::vector<double> u(N);
        for(size_t i=0; i<N; i++){
          u[i] = (aux + (double)i) / (double)N;
        }
        size_t j = 0; 
        for(size_t i=0; i<N; i++){
          while(u[i]>dist[j]){ 
            j = j+1; 
          }
          N_sons[j] = N_sons[j]+1; 
        }
        std::vector<int> zero2k_ = zero2n<int>(k);
        Eigen::VectorXi zero2k = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(zero2k_.data(), k);
        // Eigen::VectorXi zero2k = Eigen::VectorXi::LinSpaced(k, 0, k-1);
        // std::cout << zero2k << std::endl;

        Rcpp::Rcout << "zero2k " << zero2k << std::endl;
        
        Eigen::VectorXi JJ0(k+Dim);
        JJ0 << zero2k, CVec;
//        std::cout << JJ0 << std::endl;
        Eigen::VectorXi JJ = cppunique(JJ0);
//        std::cout << JJ << std::endl;
//        std::cout << "_" << std::endl;
        Eigen::MatrixXd REJJ(JJ.size(),Esum(re-1)); // pas re colonnes !!!!!!!!!!
        for(int jj=0; jj<JJ.size(); jj++){
          REJJ.row(jj) = RE.row(JJ(jj));
        }
        std::vector<Eigen::MatrixXd> ZZ(re);
        std::vector<size_t> VCVC(N,0);
        std::vector<Eigen::MatrixXi> CCCC(N);
        std::vector<Eigen::MatrixXd> VTVT(N);
        for(size_t i=0; i<N; i++){
          Rcpp::Rcout << "N_sons[i] " << N_sons[i] << std::endl;
          if(N_sons[i]){
            std::vector<size_t> VCtemp(N_sons[i],VC[i]);
            std::vector<Eigen::MatrixXd> Ztemp(re);
            // std::vector<Eigen::MatrixXd> VTtemp(N_sons[i]);
            size_t copy = N_sons[i]-1;
            Eigen::MatrixXd VTi = VT[i];
            std::vector<Eigen::MatrixXd> VTtemp(N_sons[i], VTi);
            // for(size_t ii=0; ii<N_sons[i]; ii++){
            //   VTtemp[ii].resize(Dim, VC[i]);
            // }
            for(size_t ii=0; ii<re; ii++){
              Ztemp[ii].resize(E(ii), N_sons[i]);
              
              //Rcpp::Rcout << "nrow Z[ii]" << Z[ii].rows() << std::endl;
              
              for(size_t j=0; j<N_sons[i]; j++){
                Ztemp[ii].col(j) = Z[ii].col(i);
              }
            }			
            if(copy){
              std::vector<size_t> ord = sample_int(re);
              for(size_t j=0; j<re; j++){
                size_t kk = ord[j];
                for(size_t ii=0; ii<copy; ii++){
                  // Eigen::MatrixXd XX0(n, 0);
                  Eigen::MatrixXd XX(JJ.size(), 0);
                  for(size_t jj=0; jj<re; jj++){
                    if(jj != kk){
                      // XX0.conservativeResize(Eigen::NoChange, XX0.cols()+1);
                      XX.conservativeResize(Eigen::NoChange, XX.cols()+1);
                      // XX0.rightCols(1) = RE.block(0, Esum(jj)-E(jj), n, E(jj)) *
                      //   Ztemp[jj].col(ii);
                      XX.rightCols(1) = REJJ.block(0, Esum(jj)-E(jj), JJ.size(), E(jj)) *
                        Ztemp[jj].col(ii);
                    }
                  }
                  //Eigen::MatrixXd XX(JJ.size(),re-1);
                  Eigen::VectorXi v(JJ.size());
                  for(int jj=0; jj<JJ.size(); jj++){
                    //XX.row(jj) = XX0.row(JJ(jj));
                    v(jj) = RE2(JJ(jj),kk);
                  }
                  Eigen::VectorXi vv = cppunique(v);
                  Eigen::VectorXd Z1_(vv.size());
                  for(int jj=0; jj<vv.size(); jj++){
                    Z1_(jj) = Ztemp[kk](vv(jj),ii); 
                  }
                  Eigen::MatrixXd CO2_ = REJJ.block(0, Esum(kk)-E(kk), JJ.size(), E(kk));
                  std::vector<double> colsumsCO2(E(kk));
                  std::vector<int> pcolsums;
                  for(int jj=0; jj<E(kk); jj++){
                    double colsum = Vsum(CO2_.col(jj));
                    if(colsum > 0){
                      pcolsums.push_back(jj);
                    }
                  }
                  Eigen::MatrixXd CO2(JJ.size(), pcolsums.size());
                  for(size_t jj=0; jj<pcolsums.size(); jj++){
                    CO2.col(jj) = CO2_.col(pcolsums[jj]);
                  }
                  std::vector<int> Z00;
                  for(int jj=0; jj<Z1_.size(); jj++){
                    if(Z1_(jj) != 0){
                      Z00.push_back(jj);
                    }
                  }
                  Eigen::VectorXd Z1(Z00.size());
                  for(size_t jj=0; jj<Z00.size(); jj++){
                    Z1(jj) = Z1_(Z00[jj]);
                  }
                  Eigen::MatrixXd XXX(JJ.size(), Dim-1);
                  if(fe>0){
                    Rcpp::Rcout << "JJ " << JJ << std::endl;
                    Eigen::MatrixXd FEJJ(JJ.size(),fe); // fais-le dès que tu déf JJ
                    for(int jj=0; jj<JJ.size(); jj++){
                      FEJJ.row(jj) = FE.row(JJ(jj));
                    }
                    // Rcpp::Rcout << "ncol XX " << XX.cols() << std::endl;
                    // Rcpp::Rcout << "nrow XX " << XX.rows() << std::endl;
                    // XX.conservativeResize(Eigen::NoChange, Dim-1);
                    // Rcpp::Rcout << "ncol XX " << XX.cols() << std::endl;
                    // for(size_t jj=0; jj<fe; jj++){
                    //   XX.col(re-1+jj) = FEJJ.col(jj); // cbind(FE,XX) "normalement"
                    // }
                    XXX << FEJJ, XX;
                  }
                  Rcpp::Rcout << "XXX done" << std::endl;
                  Eigen::MatrixXd MAT(JJ.size(),((int)Dim)-1+CO2.cols());
                  MAT << -XXX, CO2;
                  Rcpp::Rcout << "ok" << std::endl;
                  Rcpp::List kern = nullSpace(MAT);
                  Rcpp::Rcout << "nullspace ok" << std::endl;
                  int rk = kern["rank"];
                  Rcpp::Rcout << "ok0" << std::endl;
                  if(rk < MAT.cols()){
                    Rcpp::Rcout << "ok1" << std::endl;
                    Eigen::MatrixXd NUL = kern["kernel"];
                    Eigen::MatrixXd n1 = NUL.topRows(NUL.rows()-CO2.cols());
                    Eigen::MatrixXd n2 = NUL.bottomRows(CO2.cols());
                    Rcpp::Rcout << "ok2" << std::endl;
                    Rcpp::List QR = QRdecomp(n2);
                    Rcpp::Rcout << "ok3" << std::endl;
                    Eigen::MatrixXd O2 = QR["Q"];
                    Eigen::MatrixXd R = QR["R"];
                    // Eigen::MatrixXd O1 = tsolveAndMultiply(R, n1);
                    // Eigen::VectorXd a = O2.transpose() * Z1;
                    // Eigen::VectorXd tau_ = Z1 - (O2 * a);
                    // double b = sqrt(tau_.dot(tau_));
                    // Eigen::VectorXd tau = tau/b;

                  }
                    

                }
              }
            }

                            
          }
        }

        weight[re-1] = Eigen::MatrixXd::Ones(E(re-1),N);
      }
      
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("VERTEX") = WTnorm(0),
                            Rcpp::Named("WEIGHT") = WTnorm(0));
}