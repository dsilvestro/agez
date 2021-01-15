m = 12.5
s = 0.25



set.seed(4321)

# # read data table
# f = "/Users/dsilvestro/Software/experimental_code/AgeZirconModel/Tatacoa Geochron data data.csv"
# tbl = read.csv(f)

# PLOT ZIRCON RAW DATA AND ESTIMATED AGE
plot_data <-function(m, s, indx=c(1:10), xlim=c(9, 20), save_pdf=FALSE){
	if (save_pdf != ""){
		pdf(file="/Users/dsilvestro/Software/agez/tex/figs/ZirconAge.pdf", 6.5, 4.5)
	}
	plot(NA, xlim=xlim, ylim=c(0,1), xlab="Ma", ylab="Relative density",
	     main="", las=1) # Sampled zircon ages
	for (i in indx){
		vec = seq(m[i]-2*s[i], m[i]+2*s[i], length.out=100)
		Ys = dnorm(vec, m[i], s[i])
		Ys = Ys - min(Ys)
		Ys = Ys / max(Ys)
		lines(vec, Ys)
		polygon(c(min(vec), vec, max(vec)), c(0,Ys,0), col="#d9d9d9",border = NA) 
		segments(m[i],0,m[i],1, col="black",lty=2) # mean
		points(m[i]+rnorm(1,0,s[i]), 0, col="#e41a1c",pch=16) # mean
	}
	if (save_pdf != ""){
		dev.off()
	}
	
}

# plot_data(tbl$Age, tbl$Error.2s, indx=c(1, 10, 1000)) #, xlim=c(0,100))
plot_data(c(10, 14, 17), c(0.2, 0.4, 1), indx=c(1, 2,3), save_pdf=T) #, xlim=c(0,100))



# PLOT CAUCHY
plot_sample <- function(samples,xlim=c(0,20), UniCau=FALSE, p_error=0.1, save_pdf="",include_pI=TRUE){
	if (save_pdf != ""){
		pdf(file="/Users/dsilvestro/Software/agez/tex/figs/SampleProb.pdf", 4.5, 8)
	}
	plot(NA, xlim=xlim, ylim=c(0,0.95*length(samples)), xlab="Ma", ylab="Samples",
	     main="", yaxt='n') # "Estimated sample ages
	axis(2, labels= c(0:length(samples)-1), at=c(0:length(samples)-1), lty=, col=, las=1)
		 
	maxY = dcauchy(0,0,1.2)
	
	
	for (i in 1:length(samples)){
		samp = samples[[i]]
		vec = seq(min(xlim), max(xlim), length.out=1000)
		Ys = dcauchy(vec, samp$x, samp$s)
		if (UniCau){
			Ys[vec < samp$x] = 1/(samp$x / 2)
		}
		if (include_pI){
			# include P(I)
			Ys[vec < samp$x] = p_error * Ys[vec < samp$x]
			Ys[vec > samp$x] = (1 - p_error) * Ys[vec > samp$x]
		}
		# Ys = Ys - min(Ys)
		#Ys = 0.8 * (Ys / max(Ys))
		Ys = Ys / maxY 
		Ys = Ys + (i-1)
		lines(vec, Ys, lwd=2, col='black')
		
		polygon(c(min(xlim), vec, max(xlim)), c(i-1,Ys,i-1), col="#d9d9d9",border = NA) 
		
		abline(h=(i-1), lty=2)
		# segments(samp$x,min(Ys),samp$x,max(Ys), col="gray50",lty=2) # mean
		points(samp$z[samp$I == 1], rep((i-1), sum(samp$I)), col="#e41a1c",pch=21, bg="#e41a1c") # true
		points(samp$z[samp$I == 0], rep((i-1), length(samp$I)-sum(samp$I)),
			col="#e41a1c", bg="white",pch=21) # wrong
		points(samp$x, (i-1), col="#377eb8",pch=15) # mean
		
	}
	if (save_pdf != ""){
		dev.off()
	}
	
}




sample_1 = NULL
sample_1$x = 17
sample_1$s = 2
sample_1$z = c(12, 17.5, 18, 20)
sample_1$I = c(0, 1, 1, 1)


sample_2 = NULL
sample_2$x = 16.5
sample_2$s = 4
sample_2$z = c(20, 24)
sample_2$I = c(1, 1)


sample_3 = NULL
sample_3$x = 14
sample_3$s = 1.4
sample_3$z = c(14.5, 14.8, 15.5)
sample_3$I = c(1, 1, 1)

sample_4 = NULL
sample_4$x = 13
sample_4$s = 3
sample_4$z = c(10, 13.8, 17)
sample_4$I = c(0, 1, 1)


samples=list(sample_1, sample_2, sample_3, sample_4)

# plot_sample(samples, xlim=c(7.5,25), UniCau=F, p_error=0.1)

plot_sample(samples, xlim=c(7.5,25), UniCau=T, p_error=0.1, save_pdf=TRUE)
