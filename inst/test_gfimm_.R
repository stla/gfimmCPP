L <- npk$yield
U <- npk$yield + 0.1
n <- nrow(npk)
FE <- as.matrix(rep(1,n))
RE <- data.frame(block = npk$block)
re <- ncol(RE)+1L 
E <- integer(re) # E[i] = number of levels of i-th random effect
E[re] <- n 
for(i in 1L:(re-1L)){
  E[i] <- nlevels(RE[,i]) 
}
RE2 <- cbind(RE, factor(1L:n)) #Adds the error effect 
RE <- NULL 
for(i in 1L:re){ # Builds an indicator RE matrix for the effects
  re_levels <- levels(RE2[,i])
  for(j in 1L:E[i]){
    temp1 <- which(RE2[,i]==re_levels[j]) 
    temp2 <- integer(n) 
    temp2[temp1] <- 1L 
    RE <- cbind(RE,temp2)
  }
} 
RE2 <- sapply(RE2, as.integer)

N <- 20L
thresh <- N/2L

gfimm_(L, U, FE, RE, RE2, E, N, thresh)
