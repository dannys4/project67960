using CairoMakie, Dates
f(x) = (35x^4 - 30x^2 + 3)/8
S(x) = cos(pi*x)
dS(x) = -pi*cos(pi*x)
T(z) = acos(z)/pi
##
fig_path = "/Users/dannys4/git-repos/project_67960/blog_post/figs/"
fig_suffix = ".png"

function saveFigure(name, fig)
    save(joinpath(fig_path, name*"_"*string(now())*fig_suffix), fig)
end

theme = theme_minimal()
set_theme!(theme)
update_theme!(linewidth=3, fontsize=15)
cols = Makie.wong_colors()

##
function create_transform_fig()
    xgrid = 0:0.01:1
    fig = Figure(size=(275,275), backgroundcolor = :transparent)
    T_xgrid = T.(xgrid)
    f_xgrid = f.(xgrid)
    f_S_xgrid = f.(S.(xgrid)) .* dS.(xgrid)
    ax_T = Axis(fig[2:3,1:2], aspect=1., xlabel=L"x", ylabel=L"T(x)", backgroundcolor = :transparent)
    lines!(S.(xgrid), xgrid)
    ax_orig = Axis(fig[1, 1:2], ylabel=L"f(x)", backgroundcolor = :transparent)
    lines!(S.(xgrid), f_S_xgrid, color=cols[2])
    ax_transf = Axis(fig[2:3,3], xlabel=L"f\,(T(x)\,)|\nabla T\,|", backgroundcolor = :transparent)
    lines!(f_xgrid, xgrid, color=cols[3])
    hidedecorations!.([ax_T, ax_orig, ax_transf], label=false)
    fig
end
fig_transform = create_transform_fig()
saveFigure("transform", fig_transform)
fig_transform