function exact_value(sol::ESCHERSolver, h)
    game = sol.game
    p = player(game, h)
    if isterminal(game, h)
        return utility(game, 1, h)
    elseif iszero(p)
        A = chance_actions(game, h)
        v = 0.0
        for a in A
            v += exact_value(sol, next_hist(game, h, a))
        end
        return v/length(A)
    else
        I = vectorized_info(game, infokey(game, h))
        σ = regret_match_strategy(sol, p, I)
        A = actions(game, h)
        v = 0.0
        for i in eachindex(σ, A)
            v += σ[i]*exact_value(sol, next_hist(game, h, A[i]))
        end
        return v
    end
end

function exact_action_values(sol::ESCHERSolver, h)
    A = actions(sol.game, h)
    q = zeros(length(A))
    for i in eachindex(A)
        q[i] = exact_value(sol, next_hist(sol.game, h, A[i]))
    end
    return q
end
