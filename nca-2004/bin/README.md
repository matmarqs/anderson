To generate cix file:
  Give number of baths ...  3
  Give degeneracy of bath number 0 ...  6
  Give degeneracy of bath number 1 ...  6
  Give degeneracy of bath number 2 ...  2
  Enter your choice in format [xx-yy]+[,xx]+ (Example: 0-12,15,20)
     19-82
  Give output filename?
     cix.dat
  Output written on cix.dat

The file cix-oca.dat (by generate-oca) is equal to the example file cix4567-pc.cix.
Because of this, the directories `output-4567` and `output-oca` are exactly equal.

command to run:
```
./nca "out=./output" Sig=start/Sigma.000 Ac=start/Ac.12 cix=start/cix4567.cix U=4 T=0.1 "Ed={-20.6665403945682,-20.5769153075867,-20.6411607517974}"
```

Although output-2004 is not equal to output-4567 and output-oca, the plots of gloc.out are identical.
The conclusion must be that all `generate` executables are compatible.
