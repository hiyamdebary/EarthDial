#!/bin/bash
#
# Set this to the repo you want to clone in format [HOST/]OWNER/REPO
export TARGET_REPO=github.ibm.com/M-Ellis/going-colossal

# This will be the name of the key you are adding to your github repository
export DEPLOY_KEY_NAME=sshdeploykey-for-fms

# This will be the name of a new secret in your namespace
export SECRET_NAME="mellis-ssh-going-colossal"

# This will be the name of a new configmap in your namespace
export CONFIGMAP_NAME="github-ibm-known-hosts"

# Temporary ssh keys and host files will be termporary stored here.
# The whole directory will be removed on uninstall.
export TEMP_DIR=./temp-going-colossal

