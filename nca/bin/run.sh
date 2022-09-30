#!/bin/sh

[ -z "$1" ] && echo "Usage: ./run.sh [ACFILE] [SIGINP]" && exit 1
ACFILE=$(realpath "$1")
[ -n "$2" ] && SIGINP=$(realpath "$2")
HERE=$(dirname "$0")

if [ -n "$SIGINP" ]; then   # if there is SIGINP variable
   if [ -f "$HERE"/sig.inp ]; then  # if there is already a sig.inp, backup it
      [ ! -d "$HERE"/backup_sig ] && mkdir -p "$HERE"/backup_sig
      cp "$HERE"/sig.inp "$HERE/backup_sig/sig-$(date "+%b%d_%H:%M:%S").inp.bak"
   fi
   cp "$SIGINP" "$HERE"/sig.inp  # copy $SIGINP to ./sig.inp to be runned
else  # if there is NOT SIGINP variable
   [ ! -f "$HERE"/sig.inp ] && echo "There is not a sig.inp file and you did not specify it as \"\$2\"" && exit 1
fi

# Run the program
EXIT=0
[ ! -d "$HERE"/output ] && mkdir -p "$HERE"/output # output directory
if "$HERE"/nca "Ac=$(realpath --relative-to="$(pwd)" "$ACFILE")" > output/nca.log ; then
   # Organize the output
   mv "$HERE"/Aloc.imp "$HERE"/output
   mv "$HERE"/*.out "$HERE"/output  # gloc.out and sig.out
else
   echo "There was an error of execution"
   EXIT=1
fi

# cleaning garbage
mkdir -p "$HERE"/output/Sigma_log "$HERE"/output/Spec_log
[ -f "$(find "$HERE" -maxdepth 1 -name "Sigma.0*" | sed 1q)" ] && mv "$HERE"/Sigma.0* "$HERE"/output/Sigma_log
[ -f "$(find "$HERE" -maxdepth 1 -name "Spec.0*" | sed 1q)" ] && mv "$HERE"/Spec.0* "$HERE"/output/Spec_log
[ -f "$HERE/cores.dat" ] && mv "$HERE"/cores.dat "$HERE"/output
[ -f "$HERE/history.nca" ] && cat "$HERE"/history.nca >> "$HERE"/output/history.nca && rm "$HERE"/history.nca

if [ "$EXIT" = 0 ]; then
   exit 0
else
   exit 1
fi
