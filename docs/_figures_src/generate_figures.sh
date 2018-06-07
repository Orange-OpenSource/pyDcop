#!/bin/sh

pdflatex  -interaction=nonstopmode  factor_graph.tex   
gs -dNOPAUSE -r400 -dGraphicsAlphaBits=4 -dTextAlphaBits=4 -sDEVICE=png16m -sOutputFile=factor_graph.png -dBATCH factor_graph.pdf
cp factor_graph.png ../implementation/

pdflatex  -interaction=nonstopmode  fg_distribution.tex   
gs -dNOPAUSE -r400 -dGraphicsAlphaBits=4 -dTextAlphaBits=4 -sDEVICE=png16m -sOutputFile=fg_distribution.png -dBATCH fg_distribution.pdf
cp fg_distribution.png ../implementation/

