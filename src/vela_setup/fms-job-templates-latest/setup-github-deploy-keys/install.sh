#!/bin/bash
source vars.sh
mkdir -p $TEMP_DIR
ssh-keygen -t rsa -b 4096 -f ${TEMP_DIR}/id_rsa -P "" -N "" # Step (3)
ls $TEMP_DIR
# from $ ls $TEMP_DIR You should see
# id_rsa          id_rsa.pub
gh repo deploy-key add ${TEMP_DIR}/id_rsa.pub --title $DEPLOY_KEY_NAME --repo $TARGET_REPO
# you should see something like
# ✓ Deploy key added to $TARGET_REPO
# Next, the private ssh-key gets stored in a secret on the cluster
oc create secret generic $SECRET_NAME --from-file=${TEMP_DIR}/id_rsa
# undo oc delete secret $SECRET_NAME
ssh-keyscan github.com > ${TEMP_DIR}/known_hosts && ssh-keyscan github.ibm.com >> ${TEMP_DIR}/known_hosts
oc create configmap $CONFIGMAP_NAME --from-file=${TEMP_DIR}/known_hosts
cat << EOF

Now start the launch workflow, details are here:
https://github.ibm.com/ai-foundation/foundation-model-stack/tree/main/tools/scripts/appwrapper-pytorchjob#workflow

and add the following 3 lines to your "user-file.yaml":

sshGitCloneConfig:
    secretName: $SECRET_NAME
    configMapName: $CONFIGMAP_NAME

EOF
