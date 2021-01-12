m = 12.5
s = 0.25



set.seed(4321)

# read data table
f = "/Users/dsilvestro/Software/experimental_code/AgeZirconModel/Tatacoa Geochron data data.csv"
tbl = read.csv(f)

# PLOT ZIRCON RAW DATA AND ESTIMATED AGE
plot_data <-function(m, s, indx=c(1:10), xlim=c(9, 20)){
	plot(NA, xlim=xlim, ylim=c(0,1), xlab="Ma", ylab="Relative density",
	     main="Sampled zircon ages")
	for (i in indx){
		vec = seq(m[i]-2*s[i], m[i]+2*s[i], length.out=100)
		Ys = dnorm(vec, m[i], s[i])
		Ys = Ys - min(Ys)
		Ys = Ys / max(Ys)
		lines(vec, Ys)
		segments(m[i],0,m[i],1, col="gray50",lty=2) # mean
		points(m[i]+rnorm(1,0,s[i]), 0, col="red",pch=16) # mean
		
		
	}
}

# plot_data(tbl$Age, tbl$Error.2s, indx=c(1, 10, 1000)) #, xlim=c(0,100))
plot_data(c(10, 14, 17), c(0.2, 0.4, 1), indx=c(1, 2,3)) #, xlim=c(0,100))


# PLOT CAUCHY
plot_sample <- function(samples,xlim=c(0,20), UniCau=FALSE, p_error=0.1){
	plot(NA, xlim=xlim, ylim=c(0,length(samples)), xlab="Ma", ylab="Samples",
	     main="Estimated sample ages")
	for (i in 1:length(samples)){
		samp = samples[[i]]
		vec = seq(min(xlim), max(xlim), length.out=1000)
		Ys = dcauchy(vec, samp$x, samp$s)
		if (UniCau){
			Ys[vec < samp$x] = 1/(samp$x / 2)
			# include P(I)
			Ys[vec < samp$x] = p_error * Ys[vec < samp$x]
			Ys[vec > samp$x] = (1 - p_error) * Ys[vec > samp$x]
		}
		Ys = Ys - min(Ys)
		Ys = 0.8 * (Ys / max(Ys)) + (i-1)
		lines(vec, Ys, lwd=2, col='black')
		abline(h=min(Ys), lty=2)
		# segments(samp$x,min(Ys),samp$x,max(Ys), col="gray50",lty=2) # mean
		points(samp$z[samp$I == 1], rep(min(Ys), sum(samp$I)), col="red",pch=21, bg="red") # true
		points(samp$z[samp$I == 0], rep(min(Ys), length(samp$I)-sum(samp$I)),
			col="red", bg="white",pch=21) # wrong
		points(samp$x, min(Ys), col="purple",pch=15) # mean
		
	}
}




sample_1 = NULL
sample_1$x = 17
sample_1$s = 2
sample_1$z = c(12, 17.5, 18, 20)
sample_1$I = c(0, 1, 1, 1)

sample_2 = NULL
sample_2$x = 14
sample_2$s = 1
sample_2$z = c(14.5, 14.8, 15.5)
sample_2$I = c(1, 1, 1)

sample_3 = NULL
sample_3$x = 13
sample_3$s = 5
sample_3$z = c(10, 13.8, 17)
sample_3$I = c(0, 1, 1)


samples=list(sample_1, sample_2, sample_3)

plot_sample(samples, xlim=c(5,25), UniCau=T, p_error=0.1)
