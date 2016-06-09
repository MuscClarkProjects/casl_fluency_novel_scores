using JSON
using Lazy

typealias StringIntMap Dict{ASCIIString, Int64}
typealias JointFrequency Dict{ASCIIString, StringIntMap}

typealias StringFloatMap Dict{ASCIIString, Float64}
typealias JointProbability Dict{ASCIIString, StringFloatMap}


jointFrequencies(context_vectors_path::AbstractString) = JSON.parsefile(
  context_vectors_path, dicttype=JointFrequency)


rawProb(unigramprob_f::AbstractString) = rawProb(
  JSON.parsefile(unigramprob_f, dicttype=StringFloatMap))


jointProbabilities(context_vectors_path::AbstractString) = jointProbabilities(
  jointFrequencies(context_vectors_path))


function jointProbabilities(joint_freq::JointFrequency)
  joint_prob = JointProbability()

  for (target, context_counts) in joint_freq
    sum_contexts = context_counts |> values |> sum
    joint_prob[target] = StringFloatMap()
    for (context, context_count) in context_counts
      joint_prob[target][context] = context_count/sum_contexts
    end
  end

  joint_prob
end


function pPmi(context_vectors_path::AbstractString, logunigrams_f::AbstractString)
  pPmi(jointProbabilities(context_vectors_path), rawProb(logunigrams_f))
end


function pPmi(joint_prob::JointProbability, raw_prob::StringFloatMap)
  # Using formula p_pmi = max(0,log(P(context|targ)/P(context)))
  p_pmi = JointProbability()
  for (target::ASCIIString, context_probs::StringFloatMap) in joint_prob
    p_pmi[target] = StringFloatMap()
    for (context::ASCIIString, p_con_given_target::Float64) in context_probs
      if !haskey(raw_prob, context)
        continue
      end

      p_con = raw_prob[context]
      ppmi = p_con_given_target/p_con |> log
      
      p_pmi[target][context] = max(0., ppmi)
    end
  end

  p_pmi
end


function cosim(h1::StringFloatMap, h2::StringFloatMap)
    all_keys = union(keys(h1), keys(h2)) |> unique |> sort
    num_keys = length(all_keys)
    v1 = zeros(Float64, num_keys)
    v2 = zeros(Float64, num_keys)
    for (i, k) in enumerate(all_keys)
      v1[i] = get(h1, k, 0.)
      v2[i] = get(h2, k, 0.)
    end
    n1 = norm(v1)
    n2 = norm(v2)

    (n1 == 0. || n2 == 0.) ? 0.0 : dot(v1/n1, v2/n2)
end


function topItems(s::ASCIIString, p_pmi::JointProbability)
  cosims::Array{Tuple{Float64, ASCIIString}} = @>> keys(p_pmi) begin
    filter(k -> k != s)
    map(k -> (cosim(p_pmi[k], p_pmi[s]), k))
    sort(rev=true)
  end
  cosims[end - 15: end]
end


function simpleRun(p_pmi::JointProbability)
  for word in [ "eagle", "socks", "jacket", "asparagus", "brussels_sprouts",
                "spaghetti", "lion", "fun", "arrive", "straight" ]
    println("$(word)")
    top15 = topItems(word, p_pmi)
    for (cosim::Float64, cosim_word::ASCIIString) in top15
      println("\t $(cosim_word) $cosim")
    end
  end
end
