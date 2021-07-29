#!/usr/bin/env bash

# adapted from https://gitlab.com/mlpds_mit/ASKCOS/askcos-data/-/blob/dev/get-extra-models.sh
set -e  # exit with nonzero exit code if anything fails

usage() {
  echo
  echo "Utility script for downloading extra ASKCOS models"
  echo
  echo "Arguments:"
  echo "    -u,--username   username for authenticating to server"
  echo "    -p,--password   password for authenticating to server"
  echo "    -f,--force      download and overwrite existing files"
  echo "    -h,--help       show help"
  echo
}

# Default argument values
USERNAME=""
PASSWORD=""
FORCE=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit
      ;;
    -u|--username)
      USERNAME=$2
      shift 2
      ;;
    -p|--password)
      PASSWORD=$2
      shift 2
      ;;
    -f|--force)
      FORCE=true
      shift 1
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*) # any other flag
      echo "Error: Unsupported flag $1" >&2  # print to stderr
      usage
      exit 1
      ;;
    *) # ignore positional arguments
      shift
      ;;
  esac
done

download() {
  source=$1
  targetdir=$2
  checkdir=$3
  if [ -n "$checkdir" ] && [ -e "$checkdir" ]; then
    if [ "$FORCE" == 'false' ]; then
      echo "Skipping $checkdir because it already exists."
      return
    else
      echo "Overwriting $checkdir because it already exists and --force was specified."
    fi
  else
    echo "Downloading $checkdir ..."
  fi
  mkdir -p "$targetdir"
  (cd "$targetdir" && wget "$source")
}

if [ -z "$USERNAME" ]; then
  read -rp 'Username: ' USERNAME
fi
if [ -z "$PASSWORD" ]; then
  read -srp 'Password: ' PASSWORD
  echo
  echo
fi

SERVER="https://$USERNAME:$PASSWORD@askcos.mit.edu/files/"
DATADIR="checkpoints"

# Download context recommendation fingerprint models
download "${SERVER}models/openretro-model-archives/neuralsym_50k.mar" \
         "${DATADIR}/" \
         "${DATADIR}/neuralsym_50k.mar"

download "${SERVER}models/openretro-model-archives/gln_50k_untyped.mar" \
         "${DATADIR}/" \
         "${DATADIR}/gln_50k_untyped.mar"

download "${SERVER}models/openretro-model-archives/retroxpert_uspto50k_untyped.mar" \
         "${DATADIR}/" \
         "${DATADIR}/retroxpert_uspto50k_untyped.mar"

download "${SERVER}models/openretro-model-archives/transformer_50k_untyped.mar" \
         "${DATADIR}/" \
         "${DATADIR}/transformer_50k_untyped.mar"

echo "Done downloading all extra models!"
