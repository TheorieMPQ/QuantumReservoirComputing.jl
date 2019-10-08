mutable struct GLLayer{
    T<:Number,T1<:Number,Tm1<:AbstractMatrix{T1},
    T2<:Number,Tm2<:AbstractMatrix{T2},
    Tensalg<:DiffEqBase.BasicEnsembleAlgorithm,
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

    ntrajs::Int
    ens_alg::Tensalg
    alg::Talg
    tinput::Float64
    toutput::Float64
end

function GLLayer(tinput::Real, toutput::Real, input_size::Int,
    γ::Number, Γ::Number, g::Number, G::SimpleGraph{Int64},
    D::Number=0.0, W=missing, Win=missing; ntrajs::Int=1,
    ens_alg::DiffEqBase.BasicEnsembleAlgorithm=EnsembleSerial(),
    alg::OrdinaryDiffEq.OrdinaryDiffEqCompositeAlgorithm=AutoTsit5(Rosenbrock23())
    )
    N = nv(G)
    @assert input_size > 0
    @assert ntrajs > 0
    T = any(map(x->typeof(x)<:Complex, [γ, Γ, g, D])) ? ComplexF64 : Float64;
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

    return GLLayer{
        T,eltype(_Win),typeof(_Win),eltype(_W),typeof(_W),
        typeof(ens_alg),typeof(alg)
        }(input_size, _Win, G, _W, T(γ), T(Γ), T(g), T(D),
          ntrajs, ens_alg, alg, Float64(tinput), Float64(toutput))
end

function rand_couplings(N::Int,T::DataType)
    W = randn(T, N, N) ./ 2sqrt(N)
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
        @inbounds dψ .= m.Win * u(t)
        for v in vertices(m.G)
            Nv = neighbors(m.G, v)
            @inbounds @views dψ[v] -= im .* 10dot(m.W[v,Nv], ψ[Nv])
        end
        @. @inbounds dψ += (m.γ - (m.Γ + im*m.g)*abs2(ψ)) * ψ

        return dψ
    end

    function dψ_stoch!(dψ, ψ, p, t)
        @. @inbounds dψ = m.D

        return dψ
    end

    ψ0 = zeros(ComplexF64, nv(m.G))
    pb = SDEProblem(dψ_det!, dψ_stoch!, ψ0, tspan)
    sol = solve(EnsembleProblem(pb), m.alg, m.ens_alg; trajectories=m.ntrajs,dtmax=0.1u.Δt,save_at=0.1u.Δt)
    [sol[j].u[end][i] for i in 1:length(sol[1].u[end]), j in 1:length(sol)]
end
