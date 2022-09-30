#!/bin/sh

# This script only cleans all output and backups (BE CAREFUL)
HERE=$(dirname "$0")

rm -rf "$HERE"/backup_sig
rm -rf "$HERE"/output
