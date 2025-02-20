#!/bin/bash
source vars.sh
gh repo deploy-key delete $DEPLOY_KEY_NAME --repo $TARGET_REPO
oc delete secret $SECRET_NAME
oc delete configmap $CONFIGMAP_NAME
