#!/bin/sh
set -eu

cd "$(dirname "$0")"

if [ "$#" -ne 1 ]; then
    echo "Usage: ./docs/build.sh main_rev3[.tex]" >&2
    exit 2
fi

name=${1%.tex}
case "$name" in
    ''|*/*|*[!A-Za-z0-9_-]*)
        echo "Invalid TeX file name: $1" >&2
        exit 2
        ;;
esac

if [ ! -f "$name.tex" ]; then
    echo "TeX file not found: $name.tex" >&2
    exit 1
fi

mkdir -p "build/$name"
latexmk -pdf -synctex=1 -outdir="build/$name" "$name.tex"