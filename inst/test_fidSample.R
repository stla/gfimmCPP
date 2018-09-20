library(microbenchmark)

n <- 1000
VT2 <- rnorm(n)
VTsum <- rnorm(n)
L <- -2; U <- 2

microbenchmark(
  CPP = fidSample(VT2, VTsum, L, U),
  R = gfimm:::fid_sample(VT2, VTsum, U, L)
)
