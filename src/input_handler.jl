struct InputSignal{T<:Number}
    M::Matrix{T}
    ti::Float64
    tf::Float64
    Δt::Float64
end

function InputSignal(M::AbstractMatrix{T}, ti::Number=0.0,
                     tf::Number=ti+one(typeof(ti))) where {T<:Number}
    InputSignal{T}(Matrix(M), Float64(ti), Float64(tf), Float64(tf-ti)/size(M,2))
end

function (u::InputSignal)(t::Number)
    if u.ti <= t < u.tf
        j = Int(fld(t-u.ti, u.Δt)) + 1
        [u.M[i,j] for i in 1:size(u.M,1)]
    else
        return zeros(eltype(u.M), size(u.M,2))
    end
end
