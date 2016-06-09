using JSON
using Lazy


pluralize(w) = if w[end] == 's' || w[end] == 'z' || w[end-2:end] == "sh"
    w + "es"
  elseif w[end] == 'y'
    w[1:end-1] + "ies"
  else
    w + 's'
  end


jointFrequencies(context_vectors_path::AbstractString) = JSON.parsefile(
  context_vectors_path)


function findInflected(ks)
  println("Combining vectors for inflected targets...")
  tups = Tuple{ASCIIString, ASCIIString}[]
  for k in ks
    infl = pluralize(k)
    if infl in ks
      push!(tups, (k, infl))
  return tups
end

inflected = findInflected(filter(k, keys(joints)))
println(inflected)

function combineInflected!(hash, tups)
  for (un, infl) in tups
    for key in hash[infl]
      hash[un][key] = get(hash[un], key, 0) + hash[infl][key]
    end
  end

  return hash
end

combineInflected!(joints, inflected)


rawProb(logunigrams_f::AbstractString) = rawProp(JSON.parsefile(logunigrams_f))

function rawProb(logunigrams::Dict{ASCIIString, Float64})
  raws = [ k => exp(raw) for (k, logunigrams) in raws ]
  total_raw = raws |> values |> sum
  [key => raw/total_raw for (key, raw) in raws]
end


function contextProb(joint_freq::Dict{ASCIIString, Dict{ASCIIString, Int64}})
  context_prob = Dict()

  for (target, context_counts) in joint_freq
    sum_contexts = context_counts |> values |> sum
    context_prob[target] = Dict()
    for (context, context_count) in context_counts
      context_prob[target][context] = context_count/sum_contexts
    end
  end

  context_prob
end


function pPmi(target_context_prob, raw_prob)
  # Using formula p_pmi = max(0,log(P(context|targ)/P(context)))
  p_pmi = Dict()
  for (target, context_probs) in target_context_prob
    p_pmi[target] = {}
    for (context, context_prob) in context_probs
      p_con_given_targ = context_prob
      p_context = raw_prob[context]
      ppmi = p_con_given_targ/p_context | log
      p_pmi[target][con] = max(0, ppmi)
    end
  end

  p_pmi
end


function cosim(h1, h2)
    all_keys = union(keys(h1), keys(h2)) |> unique |> sort
    num_keys = length(all_keys)
    v1 = zeros(Float64, num_keys)
    v2 = zeros(Float64, num_keys)
    for (i,k) in enumerate(all_keys)
      v1[i] = get(h1, k, 0.)
      v2[i] = get(h2, k, 0.)
    end
    n1 = norm(v1)
    n2 = norm(v2)

    (n1 == 0. || n2 == 0.) ? 0.0 : dot(v1/n1, v2/n2)
end


function topItems(s, p_pmi)
  coses = @>> keys(p_pmi) begin
    filter(k -> k !=s )
    map(k -> (cosim(p_pmi[k], p_pmi[s]), k))
  end
  thetas = coses |> map((c, k) -> c) |> sort
  theta = thetas[end-15]

  @>> coses begin
    filter( (c, k) -> c >= theta)
    sort(rev=True)
  end
end


for word in [ "eagle", "socks", "jacket", "asparagus", "brussels_sprouts", "spaghetti", "lion", "fun", "arrive", "straight" ]
  println("$(word):")
  top15 = topItems(word)
  for (co, item) in top15
    println("\t $item $co")
  end
end
