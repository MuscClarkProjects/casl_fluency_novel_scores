immutable TwoChar
  left::Char
  right::Char

  TwoChar(left, right) = begin
    for c::Char in (left, right)
      islower(c) || error("$c must be lowercase alpha-numeric")
    end
    new(left, right)
  end
end


genUrl(tc::TwoChar, gram::Int64) = genUrl(tc.left, tc.right, gram)


Base.convert(::Type{Int64}, tc::TwoChar) = begin
  val(c::Char) = Int(c) - Int('a')
  26*val(tc.left) + val(tc.right)
end


Base.convert(::Type{TwoChar}, tc::Int64) = begin
  tc_min_a = tc - Int('a')
  left::Char = floor(Int64, tc/26) + 'a'
  right::Char = tc%26 + 'a'
  TwoChar(left, right)
end


function +(tc::TwoChar, bias::Int64)
  if sign(bias) < 0
    return tc - abs(bias)
  end
  right::Char = tc.right + bias%26
  left::Char =  tc.left + floor(Int64, bias/26)
  is_rollover::Bool = right > 'z'
  if is_rollover
    left += 1
    right -= 26
  end
  TwoChar(left, right)
end


function -(tc::TwoChar, bias::Int64)
  if sign(bias) < 0
    return tc + abs(bias)
  end
  right::Char = tc.right - bias%26
  left::Char =  tc.left - floor(Int64, abs(bias)/26)
  is_rollover::Bool = right < 'a'
  if is_rollover
    left -= 1
    right += 26
  end
  TwoChar(left, right)
end


Base.(:+)(tc1::TwoChar, tc2::TwoChar) = tc1 + Int(tc2)
Base.(:-)(tc1::TwoChar, tc2::TwoChar) = tc1 - Int(tc2)
Base.isless(tc1::TwoChar, tc2::TwoChar) = Int(tc1) < Int(tc2)
Base.one(::TwoChar) = TwoChar('a', 'b')
Base.one(::Type{TwoChar}) = one(TwoChar)
Base.zero(::TwoChar) = TwoChar('a', 'a')
Base.zero(::Type{TwoChar}) = TwoChar('a', 'a')
Base.rem(num::TwoChar, denom::TwoChar) = Int(num)%Int(denom)
Base.div(num::TwoChar, denom::TwoChar) = Int(num)/Int(denom)
Base.(:*)(i::Int64, tc::TwoChar) = TwoChar(Int(tc) * i)
Base.(:(==))(tc1::TwoChar, tc2::TwoChar) = Int(tc1) == Int(tc2)


Base.show(io::IO, tc::TwoChar) = print(io, "$(tc.left)$(tc.right)")
