PDF = final_report.pdf
TEX = final_report.tex
TEX_DEPS = references.tex reproduceable_workspace/tables/mechanism_support.tex
FIG_DEPS = \
	reproduceable_workspace/figures/yahoo_full/fig1_yahoo_smoke_popularity_shift.pdf \
	reproduceable_workspace/figures/yahoo_full/fig3_yahoo_smoke_accuracy_bias.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) $(TEX_DEPS) $(FIG_DEPS)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

clean:
	rm -f final_report.aux final_report.log final_report.out final_report.fls final_report.fdb_latexmk
