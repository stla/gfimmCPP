// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <random>
#include <vector>
// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> MatrixXl;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> VectorXl;

// [[Rcpp::export]]
Eigen::VectorXd llinsolve(const Eigen::MatrixXd & A, const Eigen::VectorXd & b){
  MatrixXl aa = A.cast<long double>();
  VectorXl bb = b.cast<long double>();
  VectorXl xx = aa.colPivHouseholderQr().solve(bb);
  Eigen::VectorXd x = xx.cast<double>();
  return x;
} 

// [[Rcpp::export]]
Eigen::MatrixXd tsolveAndMultiply(const Eigen::MatrixXd & A, const Eigen::MatrixXd & C){
  Eigen::MatrixXd M = A.triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(C);
  return M; // C %*% solve(A)
} 

// [[Rcpp::export]]
Rcpp::List nullSpace(const Eigen::MatrixXd M){
  Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
  Eigen::MatrixXd nspace = lu.kernel(); // not orthonormal
  int r = lu.rank();
  return Rcpp::List::create(Rcpp::Named("kernel") = nspace,
                            Rcpp::Named("rank") = r);
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

std::default_random_engine generator;
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
Eigen::MatrixXd fidVertex(Eigen::MatrixXd VT1, Eigen::MatrixXi CC1, 
                          Eigen::VectorXd VTsum, double L, double U, 
                          size_t Dim, int n, int k){
  size_t p = VTsum.size();
  std::vector<size_t> whichl, checkl, whichu, checku, both;
  for(size_t i=0; i<VTsum.size(); i++){
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
  size_t lcheckl = checkl.size();
  size_t lwhichl = whichl.size();
  Eigen::MatrixXi CCtemp(Dim, 0);
  Eigen::MatrixXd VTtemp(Dim, 0);
  int vert = 0;
  
  Eigen::MatrixXi CA(Dim, checkl.size());
  Eigen::MatrixXi CB(Dim, whichl.size());
  for(size_t i=0; i < Dim; i++){
    for(size_t j=0; j < lcheckl; j++){
      CA(i,j) = CC1(i, checkl[j]);
    }
    for(size_t j=0; j < whichl.size(); j++){
      CB(i,j) = CC1(i, whichl[j]);
    }
  }
  if(lcheckl < p){
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
        if(colSum == Dim-1){
          vert += 1;
          std::vector<int> which1;
          for(size_t i=0; i<Dim; i++){
            if(INT2(i,j)==1){
              which1.push_back(CB(i,ii));
            }
          }
          Eigen::VectorXi inter(which1.size()+1);
          for(size_t i=0; i<which1.size(); i++){
            inter(i) = which1[i];
          }
          inter(which1.size()) = k+n;
          Eigen::MatrixXi M;
          M = CCtemp;
          CCtemp.conservativeResize(Eigen::NoChange, CCtemp.cols()+1);
          CCtemp << M,inter; // rq: on pourrait seulement append la dernière colonne
          double lambda = (L-VTsum_wl(ii))/(VTsum_cl(j)-VTsum_wl(ii));
          Eigen::VectorXd vtnew(Dim);
          for(size_t i=0; i<Dim; i++){
            vtnew(i) = lambda*VT1_cl(i,j) + (1-lambda)*VT1_wl(i,ii);
          }
          Eigen::MatrixXd MM;
          MM = VTtemp;
          VTtemp.conservativeResize(Eigen::NoChange, VTtemp.cols()+1);
          VTtemp << MM,vtnew;
        }
      }
    }
  }
  return VTtemp;
}