function save_plots(ps)
    n = length(ps)
    for (i, p) in enumerate(ps)
        savefig(p, "/tmp/p$i.pdf")
    end
    fignames = "/tmp/p" .* string.(1:n) .* ".pdf"
    run(`pdftk $fignames cat output /tmp/all.pdf`)
end
