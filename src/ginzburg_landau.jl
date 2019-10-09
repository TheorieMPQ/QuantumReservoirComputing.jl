mutable struct GLLayer{
    T<:Number,T1<:Number,Tm1<:AbstractMatrix{T1},
    T2<:Number,Tm2<:AbstractMatrix{T2},
    Talg<:OrdinaryDiffEq.OrdinaryDiffEqCompositeAlgorithm
    }
    input_size::Int

    Win::Tm1

    G::SimpleGraph{Int64}
    W::Tm2
    γ::T
    Γ::T
    g::T
    D::T

    alg::Talg
    tinput::Float64
    toutput::Float64
end

function GLLayer(tinput::Real, toutput::Real, input_size::Int,
    γ::Number, Γ::Number, g::Number, G::SimpleGraph{Int64},
    D::Number=0.0, W=missing, Win=missing;
    alg::OrdinaryDiffEq.OrdinaryDiffEqCompositeAlgorithm=AutoTsit5(Rosenbrock23())
    )
    N = nv(G)
    @assert input_size > 0
    T = any(map(x->typeof(x)<:Union{Float32, ComplexF32}, [γ, Γ, g, D])) ? Float32 : typeof(γ)
    T = any(map(x->typeof(x)<:Complex, [γ, Γ, g, D])) ? complex(T) : T;

    _W = if ismissing(W)
        rand_couplings(N, T)
    else
        @assert size(W,1) == size(W,2) == N
        W
    end

    _Win = if ismissing(Win)
        rand_input_matrix(N, input_size, T)
    else
        @assert size(Win, 1) == N
        @assert size(Win, 2) == input_size
        Win
    end

    return GLLayer{T,eltype(_Win),typeof(_Win),eltype(_W),typeof(_W), typeof(alg)}(
          input_size, _Win, G, _W, T(γ), T(Γ), T(g), T(D),
          alg, Float64(tinput), Float64(toutput)
          )
end

function rand_couplings(N::Int,T::DataType)
    W = randn(T, N, N) ./ 2T(sqrt(N))
    for i in 1:N
        W[i,i] = zero(T)
    end
    (W .+ W') ./ 2
end

function rand_input_matrix(nr, nc, T)
    Win = rand(T, nr, nc)
    x = rand(T, nc)
    x ./= norm(x)
    Win ./= norm(Win * x)
    Win
end

function (m::GLLayer)(u::InputSignal)
    tspan = (m.tinput, m.toutput)

    function dψ_det!(dψ, ψ, p, t)
        mul!(dψ, m.Win, u(t))
        @. @inbounds dψ += (m.γ - (m.Γ + im*m.g)*abs2(ψ)) * ψ

        for v in vertices(m.G)
            Nv = neighbors(m.G, v)
            @inbounds @views dψ[v] -= im .* 10dot(m.W[v,Nv],ψ[Nv])
        end

        return dψ
    end

    function dψ_stoch!(dψ, ψ, p, t)
        @. @inbounds dψ = m.D

        return dψ
    end

    T = first(typeof(m).parameters)
    ψ0 = zeros(complex(T), nv(m.G))

    pb = SDEProblem(dψ_det!, dψ_stoch!, ψ0, tspan)
    solve(pb, m.alg; dtmax=0.1u.Δt, save_start=false, save_on=false).u[]
end

function (m::GLLayer)(u::AbstractArray{I}, args...; kwargs...) where I<:InputSignal
    tspan = (m.tinput, m.toutput)

    function dψ_det!(dψ, ψ, i, t)
        mul!(dψ, m.Win, u[i][1](t))
        @. @inbounds dψ += (m.γ - (m.Γ + im*m.g)*abs2(ψ)) * ψ

        for v in vertices(m.G)
            Nv = neighbors(m.G, v)
            @inbounds @views dψ[v] -= im .* 10dot(m.W[v,Nv],ψ[Nv])
        end
    end

    function dψ_stoch!(dψ, ψ, i, t)
        @. @inbounds dψ = m.D

        return dψ
    end

    T = first(typeof(m).parameters)
    ψ0 = zeros(complex(T), nv(m.G))

    pb = SDEProblem(dψ_det!, dψ_stoch!, ψ0, tspan, [1])
    function prob_func(prob, i, repeat)
        prob.p[1] = i
        prob
    end
    Δt = minimum(map(x->x.Δt, u))
    sol = solve(EnsembleProblem(pb, prob_func=prob_func), m.alg, args...; kwargs..., trajectories=length(u), dtmax=0.1Δt, save_start=false, save_on=false)
    reshape([sol[i].u[:][] for i in 1:length(u)], size(u))
end
