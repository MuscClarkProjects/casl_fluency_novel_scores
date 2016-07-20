casl = read.table('CASL_longitudinal_data_subjects1-78_1.0.txt',header=TRUE)

casl.nc = casl[casl$dx %in% c('nc','aami') & casl$age >= 60,]

# Want a spreadsheet that has
#   - each variable, z-transformed
#   - fci_fu at each time point
#   - summary neuropsych scores

z.transform = function(v1,v2) {
	mu = mean(v1[!is.na(v1)])
	sig = sd(v1[!is.na(v1)])

	nuvec = rep(NA,length(v2))
	if (!is.na(sig) & sig != 0) {
    	for (i in 1:length(v2)) {
	    	if (!is.na(v2[i])) {
		    	nuvec[i] = (v2[i]-mu)/sig
    		}
	    }
	} else { nuvec = v2 }
	
	return(nuvec)
}

casl.z = casl
to.transform = names(casl)[which(!(names(casl) %in% c('age','interval','sex','edu')))]
for (nom in to.transform) {
	# print(nom)
	if (is.numeric(casl[[nom]])) {
        casl.z[[nom]] = z.transform(casl.nc[[nom]],casl[[nom]])
	}
}

# Unfortunately, dom3a has no variance. Use domain totals to attain summary fci score
sum.summary = function(dfr,elnom) {
	nuvec = rep(0,length(dfr[[elnom[1]]]))
	for (i in 1:length(nuvec)) {
    	for (nom in elnom) {
    		if (!is.na(dfr[[nom]][i])) {
                nuvec[i] = nuvec[i] + dfr[[nom]][i]
            }
        }
	}
	return(nuvec)
}
# casl.z$fci = sum.summary(casl.z,c('dom2_total','dom3_total','dom4b','dom5_total','dom7_total'))
# sem = c('animal','tool','water','vehicle','boat','fruit','verb','ppt.w','ppt.p')
# casl.z$sem = sum.summary(casl.z,sem)
# casl.z$vbl = sum.summary(casl.z,c('cvlt1','cvltb','cvlt.total','cvlt.ldfr','cvlt.ldcr',sem,'ppt.w','bnt','reg','lang.1','recall1','fas','recall2','wm'))
# casl.z$exe = sum.summary(casl.z,c('fas','tmt.b','wm','pent','pent_recall'))
# casl.z$mem = sum.summary(casl.z,c('cvlt1','cvltb','cvlt.total','cvlt.ldfr','cvlt.ldcr','ten.36_1','ten.36_total','ten.36_del','recall1','recall2','pent_recall'))
# casl.z$vis = sum.summary(casl.z,c('ten.36_1','ten.36_total','ten.36_del','pent_recall','ppt.p','bnt','tmt.a','tmt.b'))
# casl.z$atn = sum.summary(casl.z,c('cvlt1','cvltb','tmt.a','ten.36_1','reg'))
# casl.4bl = casl.z # For later transformation to baseline only data

casl.z$fci = sum.summary(casl.z,c('dom2_total','dom3_total','dom4b','dom5_total','dom7_total'))
sem = c('ppt.w','ppt.p')
casl.z$sem = sum.summary(casl.z,sem)
casl.z$vbl = sum.summary(casl.z,c('cvlt1','cvltb','cvlt.total','cvlt.ldfr','cvlt.ldcr',sem,'bnt','reg','lang.1','recall1','recall2','wm'))
casl.z$exe = sum.summary(casl.z,c('tmt.b','wm','pent','pent_recall'))
casl.z$mem = sum.summary(casl.z,c('cvlt1','cvltb','cvlt.total','cvlt.ldfr','cvlt.ldcr','ten.36_1','ten.36_total','ten.36_del','recall1','recall2','pent_recall'))
casl.z$vis = sum.summary(casl.z,c('ten.36_1','ten.36_total','ten.36_del','pent_recall','ppt.p','bnt','tmt.a','tmt.b'))
casl.z$atn = sum.summary(casl.z,c('cvlt1','cvltb','tmt.a','ten.36_1','reg'))
casl.4bl = casl.z # For later transformation to baseline only data

# # Create fci_fu variable
# casl.z$fci_fu = rep(NA,nrow(casl.z))

# for (sub in unique(casl.z$subject)) {
# 	ronos = which(casl.z$subject == sub)
# 	sub.df = casl.z[ronos,]
# 	vizzes = sub.df$viscode
# 	fcis = sub.df$fci
# 	intervals = sub.df$interval
# 	dxs = casl.z$dx[ronos]
# 	dxi = which(dxs %in% c('aami','nc','mci','ad'))
# 	dxs = rep(dxs[dxi],length(ronos))
# 	if (length(vizzes) > 1) {
# 		nuviz = c(vizzes[2:length(vizzes)],vizzes[1])
# 		nufci = c(fcis[2:length(fcis)],fcis[1])
# 		nuint = c(intervals[2:length(intervals)],intervals[1])
# 		casl.z$viscode[ronos] = nuviz
# 		casl.z$fci_fu[ronos] = nufci
# 		casl.z$interval[ronos] = nuint
# 		casl.z$dx[ronos] = dxs
# 	}
# }
# casl.z = casl.z[casl.z$interval > 0,]

# # Commented out because these tables are finished
# write.table(casl.z,file="/Users/dgclark/Documents/dcfiles/manuscripts/CASL/casl_longitudinal_all_visits_8Dec2015.txt")

# # Add freesurfer gray matter values
# gm = read.table('/Users/dgclark/Documents/dcfiles/manuscripts/CASL/freesurfer_gm.txt',header=TRUE)
# for (cono in 5:76) {
# 	gm.lm = glm(gm[,cono] ~ gm$age + gm$sex + gm$eTIV)
# 	gm[,cono] = gm.lm$residuals
# }
# pc = princomp(~.,data = gm[,5:76],scores=TRUE)
# gm$age = NULL
# gm$sex = NULL
# gm$edu = NULL

# # The first PC accounts for > 45% of the variance.
# gm = cbind(gm,pc$scores[,1:19])
# gm.4bl = gm # For subsequent merge with casl.bl
# gm = merge(casl.z,gm,by='subject',all.x=TRUE)

# # Commented out because these tables are finished
# write.table(gm,file="/Users/dgclark/Documents/dcfiles/manuscripts/CASL/casl_longitudinal_all_visits_with_GM_8Dec2015.txt")

# # Compile tables for baseline only
# include.rows = which(casl.4bl$interval>0)
# fci_fu = casl.4bl$fci[include.rows]
# subject = casl.4bl$subject[include.rows]
# interval = casl.4bl$interval[include.rows]

# fci_fu.df = data.frame(subject,interval,fci_fu)
# casl.bl = casl.4bl[casl.4bl$interval==0,]
# casl.bl$interval = NULL
# casl.bl = merge(fci_fu.df,casl.bl,by='subject',all.x=TRUE)
# casl.bl.gm = merge(casl.bl,gm.4bl,by='subject',all.x=TRUE)

# write.table(casl.bl,file="/Users/dgclark/Documents/dcfiles/manuscripts/CASL/casl_longitudinal_baseline_11Dec2015.txt")
# write.table(casl.bl.gm,file="/Users/dgclark/Documents/dcfiles/manuscripts/CASL/casl_longitudinal_baseline+gm_11Dec2015.txt")
