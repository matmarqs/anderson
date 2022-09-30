#!/bin/sh

# This script only cleans all output and backups (BE CAREFUL)
HERE=$(dirname "$0")

[ -d "$HERE"/backup_sig ] && rm -rf "$HERE"/backup_sig
[ -d "$HERE"/output ] && rm -rf "$HERE"/output
