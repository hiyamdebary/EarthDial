# GitHub deploy key installation automation

Use case 1: cloning a private github.com or github.ibm.com repository into runtime container(s) on OpenShift via [GitHub deploy keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys#deploy-keys).

This automates most all of the steps for use-case 1 and provides a basis for more sophisticated automation.

## Prerequisites

The procedure uses shell scripts and the following utilities.

1. OpenShift [`oc` CLI is installed](https://docs.openshift.com/container-platform/4.11/cli_reference/openshift_cli/getting-started-cli.html#installing-openshift-cli) and you have authenticated to the target context (namespace).
2. GitHub CLI [`gh` is installed](https://github.com/cli/cli#installation) and you have logged in through `gh`.
3. Permission/access for creating deploy keys for the repository you want to clone. If you are not the owner and are unsure, follow [these steps]((https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys#deploy-keys) to find out. If you cannot create deploy keys for the repository, you may consider creating your own fork.

Mac users: see the Addendum for additional help on (1) and (2).

## Install

#### Set some variable values
Open ./vars.sh and edit the values following the comments.
```
vi vars.sh
```

#### Run the script
```
sh install.sh
```

Watch for errors, and see the next steps at the end of the output to launch workloads via the [helm charts workflow](../enqueue-job/README.md).

#### Post-Install
**Optional**:
Remove temporary files:
```
(source vars.sh && rm -rf $TEMP_DIR)
```

**Non-optional**: Keep a copy of var.sh for later reference, including for uninstalling.

## Uninstall
**Assumes** the values in vars.sh are the same ones used at install time!
```
sh uninstall.sh
```

## Reference Summary
0. [Install Homebrew for Mac](https://docs.brew.sh/Installation)
1. [Installing OpenShift oc CLI (not just for Mac)](https://docs.openshift.com/container-platform/4.11/cli_reference/openshift_cli/getting-started-cli.html#installing-openshift-cli)
2. [Installing GitHub CLI (not just for Mac)](https://github.com/cli/cli#installation)
3. [Generating a new ssh key for GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)
4. [Authoritative source on GitHub Deploy Keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys#deploy-keys)

----

## Addendum
###  Help installing the Prerequisites on Mac
Help installing the prerequisites on Mac with [Homebrew](https://docs.brew.sh/Installation):
```
# Install oc
brew install openshift-cli

# the oc login command can be copied from your openshift web console

# Install gh
brew install gh

# Login with gh
gh auth login
# follow the prompts from here
```

**Example** selection for IBM Github
```
$ gh auth login
? What account do you want to log into? GitHub Enterprise Server
? GHE hostname: github.ibm.com
? What is your preferred protocol for Git operations? HTTPS
? Authenticate Git with your GitHub credentials? Yes
? How would you like to authenticate GitHub CLI? Login with a web browser

! First copy your one-time code: GH1T-R0K5
Press Enter to open github.ibm.com in your browser...
✓ Authentication complete.
- gh config set -h github.ibm.com git_protocol https
✓ Configured git protocol
✓ Logged in as I-Am-Brave
```
