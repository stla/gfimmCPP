set.seed(123) # 3141592 for p=128 - not sure
p <- 128L
Dim <- 3L
vt1 <- matrix(rnorm(Dim*p), nrow=Dim) -> VT1
cc1 <- matrix(sample.int(p, Dim*p, replace=TRUE), nrow=Dim) -> CC1
vtsum <- rnorm(p) -> VTsum
U <- 2; L <- -2
k <- 1L
n <- Dim*p

x <- fidVertex(VT1, CC1, VTsum, L, U, Dim, n, k)

xx <- gfimm:::fid_vertex(VT1, CC1, VTsum, U, L, Dim, k, n)$CCtemp

all(x==xx)