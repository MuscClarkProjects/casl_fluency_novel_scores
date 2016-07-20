using DataFrames
using Lazy

include("../helpers/helpers.jl")


input_f = getDataFile("step3", "input",
	"CASL_longitudinal_data_subjects1-78_1.0.txt")
casl = readtable(input_f, separator='\t')

casl_nc = begin
	is_control = map(1: size(casl, 1)) do i
		is_na = @>> [casl[i, s] for s in [:dx, :age]] map(isna) reduce(|)

		if is_na
			return false
		end

		(casl[i, :dx] in ["nc", "aami"]) & (casl[i, :age] >= 60)
	end

	casl[is_control, :]
end

# Want a spreadsheet that has
#   - each variable, z-transformed
#   - fci_fu at each time point
#   - summary neuropsych scores

function zTransform(v1, v2)
	mu = @>> v1 dropna mean
	sig = @>> v1 dropna std

	in_valid_sd = isnan(sig) | (sig == 0)

	in_valid_sd ? v2 : map(v -> (v - mu)/sig, v2)
end

casl_z = copy(casl)
to_transform = @> casl names setdiff([:age, :interval, :sex, :edu])

isNumeric{T <: Number}(arr::DataArray{T, 1}) = true
isNumeric(arr) = false

for (col in to_transform)
	if isNumeric(casl[col])
	  casl_z[col] = zTransform(casl_nc[col], casl[col])
	end
end

# Unfortunately, dom3a has no variance. Use domain totals to attain summary fci score
summary(cols) = map(1:size(casl_z, 1)) do i
	data = @> casl_z[i, cols] DataArray a->a[:] dropna
	length(data) > 0 ? sum(data) : NA
end


casl_z[:fci] = summary([:dom2_total, :dom3_total, :dom4b, :dom5_total, :dom7_total])
sem = [:ppt_w, :ppt_p]
casl_z[:sem] = summary(sem)
casl_z[:vbl] = summary([:cvlt1; :cvltb; :cvlt_total; :cvlt_ldfr; :cvlt_ldcr; sem; :bnt; :reg; :lang_1; :recall1; :recall2; :wm])
casl_z[:exe] = summary([:tmt_b, :wm, :pent, :pent_recall])
casl_z[:mem] = summary([:cvlt1, :cvltb, :cvlt_total, :cvlt_ldfr, :cvlt_ldcr, :ten_36_1, :ten_36_total, :ten_36_del, :recall1, :recall2, :pent_recall])
casl_z[:vis] = summary([:ten_36_1, :ten_36_total, :ten_36_del, :pent_recall, :ppt_p, :bnt, :tmt_a, :tmt_b])
casl_z[:atn] = summary([:cvlt1, :cvltb, :tmt_a, :ten_36_1, :reg])


writetable(getDataFile("step3", "summary_scores.csv"), casl_z)
